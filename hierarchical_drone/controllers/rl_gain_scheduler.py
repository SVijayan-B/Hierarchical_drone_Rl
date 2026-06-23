import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple


class PPOGainSchedulerPolicy(nn.Module):
    """Actor-Critic Policy Network for PPO Gain Scheduler."""

    def __init__(self, input_dim: int = 17, hidden_dim: int = 32, output_dim: int = 2):
        super().__init__()
        # Actor network (Outputs mean of position and attitude gain scales)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Critic network (Outputs state value)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor(x)
        value = self.critic(x)
        return mean, value


class RLGainScheduler:
    """PPO-based Adaptive PID Gain Scheduler trained online via PPO."""

    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        c1: float = 0.5,
        c2: float = 0.01,
        epochs: int = 5,
        batch_size: int = 32,
        std: float = 0.15
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2
        self.epochs = epochs
        self.batch_size = batch_size
        self.std = std
        self.device = torch.device("cpu")

        self.policy = PPOGainSchedulerPolicy(input_dim=17).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Buffers for online PPO training
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[float] = []

        # History log for metric visualization
        self.gain_history: List[Tuple[float, float]] = []

    def get_gain_scale(
        self,
        state_arr: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[float, float]:
        """Queries the policy and returns (gain_scale_pos, gain_scale_att) in [0.7, 1.5].

        Parameters
        ----------
        state_arr : np.ndarray
            The 17-dimensional normalized state array for the scheduler.
        deterministic : bool
            Whether to use deterministic actions (evaluation mode).

        Returns
        -------
        Tuple[float, float]
            (gain_scale_pos, gain_scale_att) scaled to [0.7, 1.5].
        """
        state_t = torch.from_numpy(state_arr.astype(np.float32)).to(self.device)

        with torch.set_grad_enabled(not deterministic):
            mean, value = self.policy(state_t)
            
            if deterministic:
                action = mean
                log_prob_val = 0.0
            else:
                dist = torch.distributions.Normal(mean, self.std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                log_prob_val = float(log_prob.item())

                # Save transition details to buffers
                self.states.append(state_arr)
                self.actions.append(action.numpy())
                self.log_probs.append(log_prob_val)
                self.values.append(float(value.item()))

            # Map continuous action in R to [0.7, 1.5] using tanh
            gain_scale = 1.1 + 0.4 * torch.tanh(action)
            gain_scale_pos = float(gain_scale[0].item())
            gain_scale_att = float(gain_scale[1].item())

        self.gain_history.append((gain_scale_pos, gain_scale_att))
        return gain_scale_pos, gain_scale_att

    def save_reward_and_done(self, reward: float, done: bool):
        """Buffers step reward and done flag for PPO updates."""
        # Only log rewards/dones if we are in training mode (saving states)
        if len(self.states) > len(self.rewards):
            self.rewards.append(reward)
            self.dones.append(float(done))

    def update_policy(self) -> float:
        """Runs a policy gradient step (PPO) at the end of an episode using collected memory."""
        if not self.states or len(self.rewards) == 0:
            self.clear_buffers()
            return 0.0

        # Sync states/rewards sizes
        n_steps = min(len(self.states), len(self.rewards))
        states = np.array(self.states[:n_steps], dtype=np.float32)
        actions = np.array(self.actions[:n_steps], dtype=np.float32)
        log_probs = np.array(self.log_probs[:n_steps], dtype=np.float32)
        rewards = self.rewards[:n_steps]
        dones = self.dones[:n_steps]
        values = self.values[:n_steps] + [0.0]  # Append terminal bootstrap value

        # Compute GAE and Returns
        advantages = []
        gae = 0.0
        for i in reversed(range(n_steps)):
            delta = rewards[i] + self.gamma * values[i + 1] * (1.0 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[i]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, self.values[:n_steps])]

        # Convert to PyTorch tensors
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        old_log_probs_t = torch.tensor(log_probs, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_loss = 0.0

        # Optimization epochs
        dataset_size = n_steps
        indices = np.arange(dataset_size)

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_indices = indices[start_idx : start_idx + self.batch_size]
                
                b_states = states_t[batch_indices]
                b_actions = actions_t[batch_indices]
                b_old_log_probs = old_log_probs_t[batch_indices]
                b_returns = returns_t[batch_indices]
                b_advantages = advantages_t[batch_indices]

                # Policy forward pass
                mean, value = self.policy(b_states)
                dist = torch.distributions.Normal(mean, self.std)

                # Compute new log probs and entropy
                new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                # PPO Clipped Objective
                ratios = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss (MSE)
                critic_loss = nn.MSELoss()(value.squeeze(-1), b_returns)

                # Total Loss
                loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()

        self.clear_buffers()
        return total_loss

    def clear_buffers(self):
        """Reset buffers."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def save(self, path: str):
        """Saves policy network state dict to disk."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        """Loads policy network weights from disk."""
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
            self.policy.eval()
            print(f"[INFO] Loaded PPO Gain Scheduler checkpoints from {path}")
