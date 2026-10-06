"""MCR-UAV Reconfiguration Trajectory Buffer and Multi-Task Storage.

This module implements the deterministic, task-isolated data infrastructure
required for MCR-UAV Meta-RL training, temporal context window extraction,
and multi-tier reconfiguration parameter logging.

Key Components:
1. Transition: Strict numerical validation of state (24), action (3), reward (1),
   delta_state (24), transition tuple x_t (52), and optional latent/reconfiguration logs.
2. EpisodeBuffer: Chronologically ordered single-episode container with rolling
   history window extraction (L=20) and causal padding masks.
3. TaskTrajectoryStore: Task-isolated, split-partitioned (TRAIN, VAL, OOD_TEST)
   multi-episode storage with deterministic retrieval and serialization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class TaskSplitEnum(str, Enum):
    """Allowed dataset split categories."""
    TRAIN = "TRAIN"
    VAL = "VAL"
    OOD_TEST = "OOD_TEST"

    @classmethod
    def from_str(cls, split_str: str) -> TaskSplitEnum:
        cleaned = split_str.strip().upper()
        if cleaned in ("TRAIN", "META_TRAIN", "TRAINING"):
            return cls.TRAIN
        elif cleaned in ("VAL", "VALIDATION", "META_VAL"):
            return cls.VAL
        elif cleaned in ("OOD", "OOD_TEST", "TEST", "META_TEST"):
            return cls.OOD_TEST
        else:
            raise ValueError(
                f"Invalid split name '{split_str}'. Allowed: TRAIN, VAL, OOD_TEST."
            )


@dataclass
class Transition:
    """Single-step transition record with strict numerical integrity validation."""

    s_t: np.ndarray          # Shape: (24,)
    a_t: np.ndarray          # Shape: (3,)
    r_t: float               # Scalar reward
    s_next: np.ndarray       # Shape: (24,)
    done: bool               # Termination flag
    task_id: str             # Parent task ID
    episode_id: int          # Parent episode index
    delta_s_t: Optional[np.ndarray] = None  # Shape: (24,)

    # Optional Meta-RL context and supervisor logging fields
    z_t: Optional[np.ndarray] = None        # Latent context (16,)
    lambda_rl: Optional[float] = None       # RL authority scale
    alpha_q: Optional[float] = None         # MPC Q weight scale
    alpha_r: Optional[float] = None         # MPC R weight scale
    alpha_p: Optional[float] = None         # PID K_P scale
    alpha_i: Optional[float] = None         # PID K_I scale
    alpha_d: Optional[float] = None         # PID K_D scale
    horizon: Optional[int] = None           # MPC prediction horizon in {10, 20, 30}

    def __post_init__(self) -> None:
        """Validate shapes and numerical sanity."""
        self.s_t = np.asarray(self.s_t, dtype=np.float32)
        self.a_t = np.asarray(self.a_t, dtype=np.float32)
        self.r_t = float(self.r_t)
        self.s_next = np.asarray(self.s_next, dtype=np.float32)
        self.done = bool(self.done)
        self.episode_id = int(self.episode_id)

        # Dimension checks
        if self.s_t.shape != (24,):
            raise ValueError(f"Expected s_t shape (24,), got {self.s_t.shape}")
        if self.a_t.shape != (3,):
            raise ValueError(f"Expected a_t shape (3,), got {self.a_t.shape}")
        if self.s_next.shape != (24,):
            raise ValueError(f"Expected s_next shape (24,), got {self.s_next.shape}")

        # Compute or validate delta_s_t
        if self.delta_s_t is None:
            self.delta_s_t = self.s_next - self.s_t
        else:
            self.delta_s_t = np.asarray(self.delta_s_t, dtype=np.float32)
            if self.delta_s_t.shape != (24,):
                raise ValueError(f"Expected delta_s_t shape (24,), got {self.delta_s_t.shape}")

        # Numerical sanity checks (reject NaN / Inf)
        self._check_finite("s_t", self.s_t)
        self._check_finite("a_t", self.a_t)
        if not np.isfinite(self.r_t):
            raise ValueError(f"Reward r_t must be finite, got {self.r_t}")
        self._check_finite("s_next", self.s_next)
        self._check_finite("delta_s_t", self.delta_s_t)

        if self.z_t is not None:
            self.z_t = np.asarray(self.z_t, dtype=np.float32)
            if self.z_t.shape != (16,):
                raise ValueError(f"Expected z_t shape (16,), got {self.z_t.shape}")
            self._check_finite("z_t", self.z_t)

        for name, val in [
            ("lambda_rl", self.lambda_rl),
            ("alpha_q", self.alpha_q),
            ("alpha_r", self.alpha_r),
            ("alpha_p", self.alpha_p),
            ("alpha_i", self.alpha_i),
            ("alpha_d", self.alpha_d),
        ]:
            if val is not None and not np.isfinite(val):
                raise ValueError(f"Parameter {name} must be finite, got {val}")

        if self.horizon is not None:
            self.horizon = int(self.horizon)
            if self.horizon not in (10, 20, 30):
                raise ValueError(f"Horizon must be in (10, 20, 30), got {self.horizon}")

    @staticmethod
    def _check_finite(name: str, arr: np.ndarray) -> None:
        if np.isnan(arr).any():
            raise ValueError(f"Field '{name}' contains NaN values; rejecting numerical corruption")
        if np.isinf(arr).any():
            raise ValueError(f"Field '{name}' contains Inf values; rejecting numerical explosion")

    def get_context_vector(self) -> np.ndarray:
        """Construct the 52-dimensional transition context vector x_t = [s_t, a_t, r_t, delta_s_t]."""
        r_arr = np.array([self.r_t], dtype=np.float32)
        x_t = np.concatenate([self.s_t, self.a_t, r_arr, self.delta_s_t], axis=0)
        assert x_t.shape == (52,), f"Context vector dimension mismatch: {x_t.shape}"
        return x_t


class EpisodeBuffer:
    """Chronologically ordered single-episode trajectory container."""

    def __init__(
        self,
        task_id: str,
        task_split: Union[str, TaskSplitEnum],
        episode_id: int,
        seed: Optional[int] = None,
        max_transitions: int = 1000,
    ) -> None:
        self.task_id = str(task_id)
        self.task_split = (
            task_split if isinstance(task_split, TaskSplitEnum) else TaskSplitEnum.from_str(task_split)
        )
        self.episode_id = int(episode_id)
        self.seed = seed
        self.max_transitions = max_transitions
        self.transitions: List[Transition] = []
        self.is_finalized = False

    def append_transition(self, transition: Transition) -> None:
        """Append a validated transition to the episode."""
        if self.is_finalized:
            raise RuntimeError("Cannot append transitions to a finalized episode")
        if transition.task_id != self.task_id:
            raise ValueError(
                f"Task ID mismatch: Episode belongs to task '{self.task_id}', "
                f"got transition with task_id '{transition.task_id}'"
            )
        if len(self.transitions) >= self.max_transitions:
            raise BufferError(
                f"Episode buffer capacity ({self.max_transitions}) exceeded for task {self.task_id}"
            )
        self.transitions.append(transition)

    def finalize_episode(self) -> None:
        """Mark episode as completed and immutable."""
        self.is_finalized = True

    def length(self) -> int:
        """Return number of transitions in the episode."""
        return len(self.transitions)

    def __len__(self) -> int:
        return len(self.transitions)

    def get_transitions(self) -> List[Transition]:
        """Return all transitions in chronological order."""
        return list(self.transitions)

    def get_history(
        self,
        t: int,
        window_size: int = 20,
        as_torch: bool = False,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], int]:
        """Extract rolling history window H_t for timestep t.
        
        Parameters
        ----------
        t : int
            Target timestep (0-indexed).
        window_size : int
            Rolling history length L (default: 20).
        as_torch : bool
            Whether to return PyTorch tensors instead of NumPy arrays.
            
        Returns
        -------
        history : np.ndarray or torch.Tensor
            Tensor of shape (L, 52).
        padding_mask : np.ndarray or torch.Tensor
            Boolean mask of shape (L,) where True indicates padded tokens to ignore.
        valid_len : int
            Number of real transitions in the window.
        """
        if len(self.transitions) == 0:
            raise ValueError("Cannot extract history from an empty episode buffer")
        if t < 0 or t >= len(self.transitions):
            raise IndexError(
                f"Timestep t={t} out of range for episode with {len(self.transitions)} transitions"
            )

        history = np.zeros((window_size, 52), dtype=np.float32)
        padding_mask = np.zeros(window_size, dtype=bool)

        if t < window_size:
            # Slices 0 to t inclusive (valid_len = t + 1)
            valid_len = t + 1
            for i in range(valid_len):
                history[i] = self.transitions[i].get_context_vector()
            # Padding starts from valid_len to window_size
            padding_mask[valid_len:] = True
        else:
            # Complete 20-step history ending at timestep t
            valid_len = window_size
            start_idx = t - window_size + 1
            for i in range(window_size):
                history[i] = self.transitions[start_idx + i].get_context_vector()
            padding_mask[:] = False

        if as_torch:
            return (
                torch.from_numpy(history),
                torch.from_numpy(padding_mask),
                valid_len,
            )
        return history, padding_mask, valid_len

    def clear(self) -> None:
        """Clear all transitions from buffer."""
        self.transitions.clear()
        self.is_finalized = False


class TaskTrajectoryStore:
    """Multi-Task trajectory store enforcing strict task and split isolation."""

    def __init__(
        self,
        max_episodes_per_task: int = 100,
        max_tasks: int = 500,
        overwrite: bool = False,
    ) -> None:
        self.max_episodes_per_task = max_episodes_per_task
        self.max_tasks = max_tasks
        self.overwrite = overwrite

        # Task storage partitioned by task_id: Dict[task_id, List[EpisodeBuffer]]
        self._tasks: Dict[str, List[EpisodeBuffer]] = {}
        # Task metadata mapping: Dict[task_id, TaskSplitEnum]
        self._task_splits: Dict[str, TaskSplitEnum] = {}

    def add_episode(self, task_id: str, episode: EpisodeBuffer) -> None:
        """Add an episode to the task store."""
        if str(task_id) != episode.task_id:
            raise ValueError(
                f"Task ID mismatch: Attempting to store episode for '{episode.task_id}' "
                f"under key '{task_id}'"
            )

        task_id = str(task_id)
        if task_id not in self._tasks:
            if len(self._tasks) >= self.max_tasks:
                if not self.overwrite:
                    raise BufferError(f"Maximum task capacity ({self.max_tasks}) exceeded")
            self._tasks[task_id] = []
            self._task_splits[task_id] = episode.task_split
        else:
            # Check split consistency
            if self._task_splits[task_id] != episode.task_split:
                raise ValueError(
                    f"Split conflict for task {task_id}: existing {self._task_splits[task_id]}, "
                    f"got {episode.task_split}"
                )

        if len(self._tasks[task_id]) >= self.max_episodes_per_task:
            if not self.overwrite:
                raise BufferError(
                    f"Maximum episodes per task ({self.max_episodes_per_task}) exceeded for task {task_id}"
                )
            self._tasks[task_id].pop(0)

        self._tasks[task_id].append(episode)

    def get_task(self, task_id: str) -> List[EpisodeBuffer]:
        """Retrieve all episodes for a specific task."""
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found in store")
        return list(self._tasks[task_id])

    def get_split(self, split: Union[str, TaskSplitEnum]) -> Dict[str, List[EpisodeBuffer]]:
        """Retrieve all tasks and episodes belonging to a specific split."""
        split_enum = split if isinstance(split, TaskSplitEnum) else TaskSplitEnum.from_str(split)
        return {
            task_id: list(episodes)
            for task_id, episodes in self._tasks.items()
            if self._task_splits[task_id] == split_enum
        }

    def num_tasks(self, split: Optional[Union[str, TaskSplitEnum]] = None) -> int:
        """Return total number of tasks, optionally filtered by split."""
        if split is None:
            return len(self._tasks)
        split_enum = split if isinstance(split, TaskSplitEnum) else TaskSplitEnum.from_str(split)
        return sum(1 for s in self._task_splits.values() if s == split_enum)

    def num_episodes(self, split: Optional[Union[str, TaskSplitEnum]] = None) -> int:
        """Return total number of episodes, optionally filtered by split."""
        if split is None:
            return sum(len(eps) for eps in self._tasks.values())
        split_enum = split if isinstance(split, TaskSplitEnum) else TaskSplitEnum.from_str(split)
        return sum(
            len(eps)
            for task_id, eps in self._tasks.items()
            if self._task_splits[task_id] == split_enum
        )

    def save(self, filepath: str) -> None:
        """Serialize trajectory store to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "max_episodes_per_task": self.max_episodes_per_task,
            "max_tasks": self.max_tasks,
            "overwrite": self.overwrite,
            "tasks": self._tasks,
            "task_splits": {k: v.value for k, v in self._task_splits.items()},
        }
        torch.save(data, filepath)

    @classmethod
    def load(cls, filepath: str) -> TaskTrajectoryStore:
        """Deserialize trajectory store from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Trajectory buffer file not found: {filepath}")
        try:
            data = torch.load(filepath, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(filepath, map_location="cpu")
        store = cls(
            max_episodes_per_task=data["max_episodes_per_task"],
            max_tasks=data["max_tasks"],
            overwrite=data["overwrite"],
        )
        store._tasks = data["tasks"]
        store._task_splits = {
            k: TaskSplitEnum(v) for k, v in data["task_splits"].items()
        }
        return store
