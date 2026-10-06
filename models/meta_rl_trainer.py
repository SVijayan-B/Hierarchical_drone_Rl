"""MCR-UAV Meta-RL Trainer.

This module implements the closed-loop optimization loop for the temporal
Transformer Context Encoder ($f_\\phi$) and the Meta-Supervisor ($g_\\theta$)
using a Policy Gradient (REINFORCE) surrogate loss formulation.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.task_generator import MetaTask, TaskSplit
from evaluation.validate_tasks import TaskValidator
from hierarchical_drone.config.settings import (
    ActionConfig,
    DroneConfig,
    SensorConfig,
    SimConfig,
    TaskConfig,
)
from hierarchical_drone.env.hierarchical_nav_env import HierarchicalNavEnv
from models.meta_supervisor import MetaSupervisor, ReconfigurationVector, ReconfigurationBounds
from models.transformer_context_encoder import TransformerContextEncoder
from models.trajectory_buffer import EpisodeBuffer, Transition, TaskSplitEnum


class TrainingMode(str, Enum):
    """Supported training modes for Meta-RL optimization ablation analysis."""
    MODE_A = "A"  # Joint training (both transformer & supervisor weights optimized)
    MODE_B = "B"  # Frozen Transformer + trainable Meta-Supervisor
    MODE_C = "C"  # Frozen Meta-Supervisor + trainable Transformer
    MODE_D = "D"  # Full joint training (alias for MODE_A)


class MetaRLTrainer:
    """Trainer class executing policy-gradient Meta-RL optimization for MCR-UAV."""

    def __init__(
        self,
        transformer: TransformerContextEncoder,
        supervisor: MetaSupervisor,
        ppo_checkpoint_path: str = "results_hierarchical/run_mlp_obs_dist/final_model.zip",
        vecnormalize_path: str = "results_hierarchical/run_mlp_obs_dist/vecnormalize.pkl",
        lr: float = 1e-3,
        gamma: float = 0.99,
        explore_std: float = 0.05,
        grad_clip: float = 1.0,
        training_mode: Union[str, TrainingMode] = TrainingMode.MODE_A,
        master_seed: int = 42,
    ) -> None:
        self.device = torch.device("cpu")
        self.transformer = transformer.to(self.device)
        self.supervisor = supervisor.to(self.device)
        self.ppo_checkpoint_path = ppo_checkpoint_path
        self.vecnormalize_path = vecnormalize_path
        self.lr = lr
        self.gamma = gamma
        self.explore_std = explore_std
        self.grad_clip = grad_clip
        self.training_mode = (
            training_mode if isinstance(training_mode, TrainingMode)
            else TrainingMode(training_mode)
        )
        self.master_seed = master_seed

        # Seed locking
        self._set_seed(self.master_seed)
        self.gui_enabled: bool = False

        # Frozen PPO Policy
        if not os.path.exists(self.ppo_checkpoint_path):
            raise FileNotFoundError(f"Frozen PPO policy not found: {self.ppo_checkpoint_path}")
        self.ppo_policy = PPO.load(self.ppo_checkpoint_path, device=self.device)

        # Configure trainable parameter groups based on training mode
        self.params = []
        if self.training_mode in (TrainingMode.MODE_A, TrainingMode.MODE_D):
            # Joint training
            self.transformer.train()
            self.supervisor.train()
            self.params.extend(list(self.transformer.parameters()))
            self.params.extend(list(self.supervisor.parameters()))
        elif self.training_mode == TrainingMode.MODE_B:
            # Frozen Transformer + trainable Supervisor
            self.transformer.eval()
            self.supervisor.train()
            for p in self.transformer.parameters():
                p.requires_grad = False
            self.params.extend(list(self.supervisor.parameters()))
        elif self.training_mode == TrainingMode.MODE_C:
            # Trainable Transformer + frozen Supervisor
            self.transformer.train()
            self.supervisor.eval()
            for p in self.supervisor.parameters():
                p.requires_grad = False
            self.params.extend(list(self.transformer.parameters()))

        self.optimizer = optim.Adam(self.params, lr=self.lr) if len(self.params) > 0 else None

    def _set_seed(self, seed: int) -> None:
        """Lock deterministic seeds across packages."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _make_env(self, task: MetaTask, episode_sec: float = 6.0, allow_ood: bool = False, gui: Optional[bool] = None) -> Tuple[DummyVecEnv, HierarchicalNavEnv]:
        """Create clean seed-locked dummy vec env wrapping HierarchicalNavEnv."""
        # Check task split guard to prevent OOD leakage
        if not allow_ood and TaskSplitEnum.from_str(task.split) == TaskSplitEnum.OOD_TEST:
            raise ValueError(
                f"[OOD Lock Guard] STRICTLY PROHIBITED: Task '{task.task_id}' has OOD_TEST split "
                f"and cannot be loaded into the training dataset."
            )

        use_gui = self.gui_enabled if gui is None else gui

        def _init() -> HierarchicalNavEnv:
            sim_cfg = SimConfig(
                gui=use_gui,
                episode_sec=episode_sec,
                pyb_freq=240,
                ctrl_freq=120,
                rl_freq=10,
            )
            task_cfg = TaskConfig(
                target_threshold_m=0.60,
                world_xy_limit=2.5,
                world_z_min=0.1,
                world_z_max=2.2,
            )
            sensor_cfg = SensorConfig()
            action_cfg = ActionConfig()

            env = HierarchicalNavEnv(
                sim=sim_cfg,
                task=task_cfg,
                sensor_cfg=sensor_cfg,
                action_cfg=action_cfg,
                use_adaptive_scheduler=True,
                demo_guided_mode=False,
                use_mpc_layer=True,
                use_rl_gain_scheduler=False,
                use_adaptive_mpc=False,
                use_history=False,
                domain_randomization=False,
            )
            env.evaluation_mode = True
            return env

        vec_env = DummyVecEnv([_init])
        vec_env = VecNormalize.load(self.vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        raw_env: HierarchicalNavEnv = vec_env.envs[0]
        TaskValidator().inject_task(raw_env, task)
        return vec_env, raw_env

    def rollout_episode(
        self,
        task: MetaTask,
        explore: bool = True,
        max_steps: Optional[int] = None,
        allow_ood: bool = False,
        telemetry_callback: Optional[Any] = None,
        control_check_callback: Optional[Any] = None,
        gui: Optional[bool] = None,
    ) -> Tuple[EpisodeBuffer, List[torch.Tensor], float]:
        """Roll out a single episode on a specific task.
        
        Returns:
            episode_buffer: Trajectory storage.
            episode_log_probs: Log probabilities of actions for backprop.
            total_reward: Total scalar return.
        """
        vec_env, raw_env = self._make_env(task, allow_ood=allow_ood, gui=gui)
        raw_env.reconfig_active = True

        # Unique episode buffer setup
        episode_buffer = EpisodeBuffer(
            task_id=task.task_id,
            task_split=task.split,
            episode_id=0,
            seed=task.seed,
        )

        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0
        episode_log_probs = []

        try:
            # First state extraction
            raw_obs = raw_env._get_obs(raw_env.last_state)[0]
            s_t = np.concatenate([raw_obs[0:21], raw_env.prev_action[0:3]], axis=0)

            while not done:
                # 1. Construct history window H_t (length 20, padding if t < 20)
                if step_count == 0:
                    history = np.zeros((20, 52), dtype=np.float32)
                    padding_mask = np.ones(20, dtype=bool)
                    padding_mask[0] = False
                    valid_len = 1
                else:
                    h_np, pm_np, v_len = episode_buffer.get_history(step_count - 1, window_size=20)
                    history = h_np
                    padding_mask = pm_np
                    valid_len = v_len

                # Convert to torch batched tensors
                history_t = torch.from_numpy(history).unsqueeze(0).to(self.device)
                padding_mask_t = torch.from_numpy(padding_mask).unsqueeze(0).to(self.device)
                valid_lens_t = torch.tensor([valid_len], dtype=torch.long, device=self.device)

                # 2. Forward pass Context Encoder & Meta-Supervisor
                with torch.set_grad_enabled(explore and (self.training_mode != TrainingMode.MODE_B)):
                    z_t = self.transformer(history_t, padding_mask_t, valid_lens_t)

                with torch.set_grad_enabled(explore and (self.training_mode != TrainingMode.MODE_C)):
                    reconfig_out: ReconfigurationVector = self.supervisor(z_t, return_hard=not explore)

                # 3. Parameter exploration or deterministic exploitation
                if explore:
                    # Continuous head exploration (Gaussian policy)
                    mu = reconfig_out.continuous_vector  # Shape: [1, 6]
                    noise = torch.randn_like(mu) * self.explore_std
                    raw_sampled = mu + noise
                    
                    # Manual sigmoid mapping matching MetaSupervisor bounds
                    # lambda_rl in [0.0, 1.0]
                    lambda_rl = torch.clamp(raw_sampled[:, 0:1], 0.0, 1.0)
                    # alpha_q, alpha_r in [0.2, 5.0]
                    alpha_q = torch.clamp(raw_sampled[:, 1:2], 0.2, 5.0)
                    alpha_r = torch.clamp(raw_sampled[:, 2:3], 0.2, 5.0)
                    # alpha_p, alpha_d in [0.5, 2.0]
                    alpha_p = torch.clamp(raw_sampled[:, 3:4], 0.5, 2.0)
                    # alpha_i in [0.2, 2.5]
                    alpha_i = torch.clamp(raw_sampled[:, 4:5], 0.2, 2.5)
                    # alpha_d scale
                    alpha_d = torch.clamp(raw_sampled[:, 5:6], 0.5, 2.0)

                    continuous_sampled = torch.cat([lambda_rl, alpha_q, alpha_r, alpha_p, alpha_i, alpha_d], dim=-1)

                    # Horizon categorical sampling
                    horizon_probs = reconfig_out.horizon_probabilities[0]  # [3]
                    dist_H = torch.distributions.Categorical(probs=horizon_probs)
                    h_idx = dist_H.sample()
                    horizon_val = raw_env.reconfig_controller.bounds.horizon_options[h_idx.item()]

                    # Log-probability calculation
                    # Gaussian log pdf
                    var = self.explore_std ** 2
                    log_prob_c = -0.5 * math.log(2.0 * math.pi * var) - ((continuous_sampled - mu) ** 2) / (2.0 * var)
                    log_prob_c = log_prob_c.sum(dim=-1)  # sum over 6 continuous dimensions

                    # Categorical log pdf
                    log_prob_h = dist_H.log_prob(h_idx)
                    total_log_prob = log_prob_c + log_prob_h
                    episode_log_probs.append(total_log_prob)

                    # Assemble command vector c_t
                    c_t = np.array([
                        lambda_rl.item(),
                        alpha_q.item(),
                        alpha_r.item(),
                        alpha_p.item(),
                        alpha_i.item(),
                        alpha_d.item(),
                        float(horizon_val)
                    ], dtype=np.float32)
                else:
                    # Deterministic evaluation using supervisor outputs
                    c_t = np.array([
                        reconfig_out.lambda_rl.item(),
                        reconfig_out.alpha_q.item(),
                        reconfig_out.alpha_r.item(),
                        reconfig_out.alpha_p.item(),
                        reconfig_out.alpha_i.item(),
                        reconfig_out.alpha_d.item(),
                        reconfig_out.horizon.item()
                    ], dtype=np.float32)

                # Apply reconfiguration parameters to env
                raw_env.apply_reconfiguration(c_t)

                # Get action from frozen PPO policy
                action, _ = self.ppo_policy.predict(obs, deterministic=True)

                # Step environment
                obs, reward, dones, infos = vec_env.step(action)
                done = dones[0]
                ep_reward += float(reward[0])

                # Get next raw observation and state transition details
                raw_obs_next = raw_env._get_obs(raw_env.last_state)[0]
                s_next = np.concatenate([raw_obs_next[0:21], raw_env.prev_action[0:3]], axis=0)

                # Formulate action history a_t (dimension 3 velocity commands)
                # Note: action indices 0:3 represent vx, vy, vz
                a_t = np.array(action[0, 0:3], dtype=np.float32)

                # Append transition to episode buffer
                trans = Transition(
                    s_t=s_t,
                    a_t=a_t,
                    r_t=float(reward[0]),
                    s_next=s_next,
                    done=done,
                    task_id=task.task_id,
                    episode_id=0,
                    z_t=z_t.detach().cpu().numpy().ravel(),
                    lambda_rl=float(c_t[0]),
                    alpha_q=float(c_t[1]),
                    alpha_r=float(c_t[2]),
                    alpha_p=float(c_t[3]),
                    alpha_i=float(c_t[4]),
                    alpha_d=float(c_t[5]),
                    horizon=int(c_t[6]),
                )
                episode_buffer.append_transition(trans)

                # Step variables update
                s_t = s_next.copy()
                step_count += 1

                if telemetry_callback is not None:
                    try:
                        telemetry_callback({
                            "step": step_count,
                            "position": raw_env.last_state[0:3].tolist(),
                            "target": raw_env.target.tolist(),
                            "velocity": raw_env.last_state[3:6].tolist(),
                            "rpy": raw_env.last_state[6:9].tolist(),
                            "motor_rpm": raw_env.last_state[12:16].tolist(),
                            "tracking_error": float(np.linalg.norm(raw_env.last_state[0:3] - raw_env.target)),
                            "reward": float(reward[0]),
                            "cum_reward": float(ep_reward),
                            "lambda_rl": float(c_t[0]),
                            "alpha_q": float(c_t[1]),
                            "alpha_r": float(c_t[2]),
                            "alpha_p": float(c_t[3]),
                            "alpha_i": float(c_t[4]),
                            "alpha_d": float(c_t[5]),
                            "horizon": int(c_t[6]),
                            "z_t": z_t.detach().cpu().numpy().ravel().tolist(),
                            "done": done,
                            "success": bool(infos[0].get("success", False)),
                        })
                    except Exception:
                        pass

                if control_check_callback is not None:
                    try:
                        if not control_check_callback():
                            break
                    except Exception:
                        pass

                if max_steps is not None and step_count >= max_steps:
                    break

            if step_count > 0:
                episode_buffer.final_info = infos[0]
            else:
                episode_buffer.final_info = {}
            episode_buffer.finalize_episode()

        finally:
            try:
                vec_env.close()
            except Exception:
                pass

        return episode_buffer, episode_log_probs, ep_reward

    def compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted cumulative return G_t."""
        returns = []
        g = 0.0
        for r in reversed(rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        return returns

    def train_step(self, tasks: List[MetaTask]) -> Dict[str, Any]:
        """Perform a single policy gradient training step across a batch of tasks."""
        if self.optimizer is None:
            # Frozen mode
            return {"loss_total": 0.0, "mean_return": 0.0}

        self.optimizer.zero_grad()

        batch_log_probs = []
        batch_returns = []
        batch_rewards = []
        all_rewards = []

        # 1. Roll out episodes for each task
        for task in tasks:
            ep_buffer, log_probs, total_reward = self.rollout_episode(task, explore=True)
            transitions = ep_buffer.get_transitions()
            
            rewards = [t.r_t for t in transitions]
            returns = self.compute_returns(rewards)

            batch_log_probs.extend(log_probs)
            batch_returns.extend(returns)
            batch_rewards.append(total_reward)
            all_rewards.extend(rewards)

        if len(batch_log_probs) == 0:
            return {"loss_total": 0.0, "mean_return": 0.0}

        # 2. Normalize returns (standard variance reduction)
        returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=self.device)
        if len(returns_t) > 1:
            mean = returns_t.mean()
            std = returns_t.std() + 1e-8
            returns_t = (returns_t - mean) / std

        # 3. Compute REINFORCE surrogate loss: L = - 1/N * sum(log_prob * normalized_return)
        loss = 0.0
        for log_prob, ret in zip(batch_log_probs, returns_t):
            loss = loss - log_prob * ret
        loss = loss / len(batch_log_probs)

        # 4. Backpropagation
        loss.backward()

        # Capture gradients before optimizer step (for gradient audit)
        grad_audit = self.audit_gradients()

        # Parameter change audit snapshot
        param_snapshot = self.get_parameter_snapshot()

        # Gradient clipping
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)

        # Optimize weights
        self.optimizer.step()

        # Parameter change audit comparison
        param_changes = self.compare_parameters(param_snapshot)

        return {
            "loss_total": loss.item(),
            "mean_return": float(np.mean(batch_rewards)),
            "mean_reward": float(np.mean(all_rewards)),
            "grad_audit": grad_audit,
            "param_changes": param_changes,
        }

    def audit_gradients(self) -> Dict[str, Dict[str, Any]]:
        """Audit gradients for each trainable parameter group."""
        audit_results = {}

        # Context Encoder Group
        if self.training_mode in (TrainingMode.MODE_A, TrainingMode.MODE_D, TrainingMode.MODE_C):
            audit_results["transformer"] = self._audit_module_grads(self.transformer)

        # Meta-Supervisor Group
        if self.training_mode in (TrainingMode.MODE_A, TrainingMode.MODE_D, TrainingMode.MODE_B):
            audit_results["meta_supervisor"] = self._audit_module_grads(self.supervisor)

        return audit_results

    def _audit_module_grads(self, module: nn.Module) -> Dict[str, Any]:
        p_count = 0
        grad_norms = []
        nonzero_grads = 0
        total_elements = 0
        nan_count = 0
        inf_count = 0

        for p in module.parameters():
            if p.requires_grad:
                p_count += p.numel()
                if p.grad is not None:
                    g = p.grad.detach().cpu().numpy()
                    grad_norms.append(float(np.linalg.norm(g)))
                    nonzero_grads += int(np.sum(g != 0.0))
                    total_elements += g.size
                    nan_count += int(np.sum(np.isnan(g)))
                    inf_count += int(np.sum(np.isinf(g)))

        mean_norm = float(np.mean(grad_norms)) if len(grad_norms) > 0 else 0.0
        fraction_nonzero = float(nonzero_grads / total_elements) if total_elements > 0 else 0.0

        return {
            "parameter_count": p_count,
            "gradient_norm": mean_norm,
            "nonzero_gradient_fraction": fraction_nonzero,
            "NaN_count": nan_count,
            "Inf_count": inf_count,
        }

    def get_parameter_snapshot(self) -> Dict[str, torch.Tensor]:
        """Get a copy of current trainable parameters."""
        snapshot = {}
        for name, param in self.transformer.named_parameters():
            if param.requires_grad:
                snapshot[f"transformer.{name}"] = param.detach().clone()
        for name, param in self.supervisor.named_parameters():
            if param.requires_grad:
                snapshot[f"supervisor.{name}"] = param.detach().clone()
        return snapshot

    def compare_parameters(self, snapshot: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compare current parameters with a snapshot to evaluate parameter deltas."""
        changed_tensors = 0
        l2_deltas = []
        max_deltas = []

        for name, param in self.transformer.named_parameters():
            if param.requires_grad:
                snap_p = snapshot.get(f"transformer.{name}")
                if snap_p is not None:
                    diff = param.detach() - snap_p
                    l2 = float(torch.norm(diff).item())
                    m = float(torch.max(torch.abs(diff)).item())
                    l2_deltas.append(l2)
                    max_deltas.append(m)
                    if m > 1e-8:
                        changed_tensors += 1

        for name, param in self.supervisor.named_parameters():
            if param.requires_grad:
                snap_p = snapshot.get(f"supervisor.{name}")
                if snap_p is not None:
                    diff = param.detach() - snap_p
                    l2 = float(torch.norm(diff).item())
                    m = float(torch.max(torch.abs(diff)).item())
                    l2_deltas.append(l2)
                    max_deltas.append(m)
                    if m > 1e-8:
                        changed_tensors += 1

        mean_l2 = float(np.mean(l2_deltas)) if len(l2_deltas) > 0 else 0.0
        max_val = float(np.max(max_deltas)) if len(max_deltas) > 0 else 0.0

        return {
            "L2_delta": mean_l2,
            "max_delta": max_val,
            "changed_tensors": changed_tensors,
        }

    def save_checkpoints(self, path: str = "results/meta_rl/", config: Optional[Dict[str, Any]] = None) -> None:
        """Serialize optimizer and model state weights."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.transformer.state_dict(), os.path.join(path, "transformer_checkpoint.pt"))
        torch.save(self.supervisor.state_dict(), os.path.join(path, "meta_supervisor_checkpoint.pt"))
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer_checkpoint.pt"))

        # Save config files
        if config is not None:
            with open(os.path.join(path, "training_config.yaml"), "w", encoding="utf-8") as f:
                for k, v in config.items():
                    f.write(f"{k}: {v}\n")

        # Save manifest info
        manifest_meta = {
            "reproducibility": {
                "master_seed": self.master_seed,
                "explore_std": self.explore_std,
                "learning_rate": self.lr,
                "training_mode": self.training_mode.value,
            }
        }
        with open(os.path.join(path, "training_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest_meta, f, indent=2)

    def load_checkpoints(self, path: str = "results/meta_rl/") -> None:
        """Deserialize optimizer and model state weights."""
        transformer_path = os.path.join(path, "transformer_checkpoint.pt")
        supervisor_path = os.path.join(path, "meta_supervisor_checkpoint.pt")
        optimizer_path = os.path.join(path, "optimizer_checkpoint.pt")

        if os.path.exists(transformer_path):
            self.transformer.load_state_dict(torch.load(transformer_path, map_location=self.device))
        if os.path.exists(supervisor_path):
            self.supervisor.load_state_dict(torch.load(supervisor_path, map_location=self.device))
        if self.optimizer is not None and os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
