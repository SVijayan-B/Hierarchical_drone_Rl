"""Deterministic, Reproducible Meta-RL Task Distribution Generator for MCR-UAV.

This module provides structured task generation for:
- Meta-Training Distribution: 80 tasks (in-distribution disturbances & faults)
- Meta-Validation Distribution: 20 tasks (interpolation across training bounds)
- Out-of-Distribution (OOD) Meta-Test Distribution: 30 tasks (severe unseen conditions)
Total: 130 deterministic tasks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class TaskSplit(str, Enum):
    """Enumeration of Meta-RL task evaluation splits."""
    TRAIN = "train"
    VALIDATION = "val"
    OOD_TEST = "ood_test"


@dataclass
class MetaTask:
    """Structured, serializable representation of a single Meta-RL task configuration.
    
    Attributes
    ----------
    task_id : str
        Unique deterministic identifier (e.g., 'task_train_001').
    seed : int
        Deterministic random seed used to generate this specific task.
    split : str
        Dataset split: 'train', 'val', or 'ood_test'.
    wind_magnitude : float
        Steady-state wind speed in m/s.
    wind_direction : List[float]
        3D normalized unit vector for wind direction [wx, wy, wz].
    gust_magnitude : float
        Peak amplitude of sinusoidal / turbulent wind gusts in m/s.
    gust_frequency : float
        Frequency of periodic wind gusts in Hz.
    turbulence_std : float
        Standard deviation of stochastic Dryden-like Gaussian wind turbulence in m/s.
    impulse_magnitude : float
        Magnitude of sudden external lateral force pulse in Newtons (N).
    impulse_duration : float
        Duration of the impulse disturbance in seconds (default 0.1s / 1 RL step).
    impulse_onset : float
        Simulation timestamp (seconds) at which impulse is injected.
    sensor_noise_scale : float
        Multiplicative scale factor applied to nominal IMU and ultrasonic noise std.
    motor_degradation : float
        Fractional thrust loss-of-effectiveness (LoE) in [0.0, 0.70].
    degraded_motors : List[int]
        Indices of degraded rotors (0-indexed: 0=front-right, 1=rear-left, 2=front-left, 3=rear-right).
    mass_scale : float
        Scale multiplier for drone nominal mass (0.027 kg nominal).
    inertia_scale : float
        Scale multiplier for drone inertia tensor diag(ixx, iyy, izz).
    waypoint_seed : int
        Seed used by global A* / trajectory generator for obstacle and goal placement.
    compound_disturbance_flags : Dict[str, bool]
        Flags indicating which disturbance modalities are actively injected.
    """

    task_id: str
    seed: int
    split: str
    wind_magnitude: float
    wind_direction: List[float]
    gust_magnitude: float
    gust_frequency: float
    turbulence_std: float
    impulse_magnitude: float
    impulse_duration: float
    impulse_onset: float
    sensor_noise_scale: float
    motor_degradation: float
    degraded_motors: List[int]
    mass_scale: float
    inertia_scale: float
    waypoint_seed: int
    compound_disturbance_flags: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task dataclass to a standard JSON/YAML-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetaTask:
        """Construct MetaTask instance from a dictionary."""
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize task to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class TaskGenerator:
    """Deterministic, seedable task generator for MCR-UAV Meta-RL protocols."""

    # Exact Phase 2 distribution budgets
    TRAIN_COUNT: int = 80
    VAL_COUNT: int = 20
    OOD_COUNT: int = 30
    TOTAL_COUNT: int = 130

    # Range boundaries
    # Train / Val ranges
    TRAIN_WIND_MAG: Tuple[float, float] = (0.0, 2.5)       # m/s
    TRAIN_GUST_MAG: Tuple[float, float] = (0.0, 1.5)       # m/s
    TRAIN_GUST_FREQ: Tuple[float, float] = (0.5, 2.0)      # Hz
    TRAIN_TURB_STD: Tuple[float, float] = (0.0, 0.15)      # m/s
    TRAIN_IMPULSE_MAG: Tuple[float, float] = (1.0, 3.0)    # N
    TRAIN_NOISE_SCALE: Tuple[float, float] = (1.0, 2.0)    # x nominal
    TRAIN_MOTOR_DEG: Tuple[float, float] = (0.0, 0.25)     # 0% to 25% LoE
    TRAIN_MASS_SCALE: Tuple[float, float] = (0.90, 1.15)
    TRAIN_INERTIA_SCALE: Tuple[float, float] = (0.90, 1.15)

    # OOD Test ranges (Strictly separated where required)
    OOD_WIND_MAG: Tuple[float, float] = (3.0, 5.5)         # m/s (> 2.5)
    OOD_GUST_MAG: Tuple[float, float] = (2.0, 4.0)         # m/s (> 1.5)
    OOD_GUST_FREQ: Tuple[float, float] = (2.5, 5.0)        # Hz (> 2.0)
    OOD_TURB_STD: Tuple[float, float] = (0.20, 0.45)       # m/s (> 0.15)
    OOD_IMPULSE_MAG: Tuple[float, float] = (4.0, 6.0)      # N (> 3.0)
    OOD_NOISE_SCALE: Tuple[float, float] = (3.5, 5.0)      # x nominal (> 2.0)
    OOD_MOTOR_DEG: Tuple[float, float] = (0.30, 0.70)      # 30% to 70% LoE (> 25%)
    OOD_MASS_SCALE: Tuple[float, float] = (1.20, 1.50)     # (> 1.15)
    OOD_INERTIA_SCALE: Tuple[float, float] = (1.20, 1.50)  # (> 1.15)

    def __init__(self, master_seed: int = 42) -> None:
        """Initialize the task generator with a master seed.
        
        Parameters
        ----------
        master_seed : int
            Master seed governing all deterministic pseudo-random task streams.
        """
        self.master_seed = master_seed

    def _sample_unit_vector_3d(self, rng: np.random.Generator) -> List[float]:
        """Sample a 3D unit vector uniformly distributed on the upper/horizontal sphere."""
        theta = rng.uniform(0.0, 2.0 * math.pi)
        phi = rng.uniform(-math.pi / 6.0, math.pi / 6.0)  # Primarily horizontal wind with slight tilt
        x = math.cos(phi) * math.cos(theta)
        y = math.cos(phi) * math.sin(theta)
        z = math.sin(phi)
        vec = np.array([x, y, z], dtype=np.float64)
        vec = vec / np.linalg.norm(vec)
        return [round(float(v), 4) for v in vec]

    def generate_train_tasks(self) -> List[MetaTask]:
        """Generate exactly 80 reproducible training tasks."""
        # Dedicated RNG stream derived from master seed + split offset
        rng = np.random.Generator(np.random.PCG64(self.master_seed + 1000))
        tasks: List[MetaTask] = []

        # Mixture composition:
        # 16 Nominal / Light Wind
        # 16 Gust / Turbulence
        # 16 Impulse perturbations
        # 16 Sensor noise attacks
        # 16 Moderate actuator degradation ([0%, 25%])
        categories = ["wind", "gust", "impulse", "sensor", "motor_deg"]

        for i in range(self.TRAIN_COUNT):
            task_seed = int(self.master_seed + 1000 + i)
            task_rng = np.random.Generator(np.random.PCG64(task_seed))
            category = categories[i % len(categories)]

            # Sample base parameters within training ranges
            wind_mag = float(task_rng.uniform(*self.TRAIN_WIND_MAG)) if category in ["wind", "gust"] else float(task_rng.uniform(0.0, 1.0))
            wind_dir = self._sample_unit_vector_3d(task_rng)
            gust_mag = float(task_rng.uniform(*self.TRAIN_GUST_MAG)) if category == "gust" else 0.0
            gust_freq = float(task_rng.uniform(*self.TRAIN_GUST_FREQ)) if category == "gust" else 1.0
            turb_std = float(task_rng.uniform(*self.TRAIN_TURB_STD)) if category == "gust" else 0.02
            
            impulse_mag = float(task_rng.uniform(*self.TRAIN_IMPULSE_MAG)) if category == "impulse" else 0.0
            impulse_dur = 0.10  # 1 RL step
            impulse_onset = round(float(task_rng.uniform(1.5, 3.5)), 2)

            noise_scale = float(task_rng.uniform(*self.TRAIN_NOISE_SCALE)) if category == "sensor" else 1.0
            motor_deg = float(task_rng.uniform(*self.TRAIN_MOTOR_DEG)) if category == "motor_deg" else 0.0
            
            # Degraded motors selection
            if motor_deg > 0.0:
                deg_choice = task_rng.choice([1, 2], p=[0.7, 0.3])
                degraded_motors = [int(m) for m in task_rng.choice([0, 1, 2, 3], size=deg_choice, replace=False)]
            else:
                degraded_motors = []

            mass_scale = round(float(task_rng.uniform(*self.TRAIN_MASS_SCALE)), 3)
            inertia_scale = round(float(task_rng.uniform(*self.TRAIN_INERTIA_SCALE)), 3)
            waypoint_seed = int(self.master_seed + 10000 + i)

            flags = {
                "has_wind": wind_mag > 0.5,
                "has_gusts": gust_mag > 0.0,
                "has_impulse": impulse_mag > 0.0,
                "has_sensor_noise": noise_scale > 1.2,
                "has_motor_degradation": motor_deg > 0.0,
                "is_compound": sum([wind_mag > 0.5, impulse_mag > 0.0, noise_scale > 1.2, motor_deg > 0.0]) > 1
            }

            task = MetaTask(
                task_id=f"train_task_{i+1:03d}",
                seed=task_seed,
                split=TaskSplit.TRAIN.value,
                wind_magnitude=round(wind_mag, 3),
                wind_direction=wind_dir,
                gust_magnitude=round(gust_mag, 3),
                gust_frequency=round(gust_freq, 3),
                turbulence_std=round(turb_std, 3),
                impulse_magnitude=round(impulse_mag, 3),
                impulse_duration=round(impulse_dur, 2),
                impulse_onset=impulse_onset,
                sensor_noise_scale=round(noise_scale, 3),
                motor_degradation=round(motor_deg, 3),
                degraded_motors=degraded_motors,
                mass_scale=mass_scale,
                inertia_scale=inertia_scale,
                waypoint_seed=waypoint_seed,
                compound_disturbance_flags=flags
            )
            tasks.append(task)

        return tasks

    def generate_validation_tasks(self) -> List[MetaTask]:
        """Generate exactly 20 validation tasks for interpolation / hyperparameter tuning."""
        rng = np.random.Generator(np.random.PCG64(self.master_seed + 2000))
        tasks: List[MetaTask] = []

        for i in range(self.VAL_COUNT):
            task_seed = int(self.master_seed + 2000 + i)
            task_rng = np.random.Generator(np.random.PCG64(task_seed))

            # Interpolation within training ranges with compound pairings
            wind_mag = float(task_rng.uniform(*self.TRAIN_WIND_MAG))
            wind_dir = self._sample_unit_vector_3d(task_rng)
            gust_mag = float(task_rng.uniform(0.0, self.TRAIN_GUST_MAG[1]))
            gust_freq = float(task_rng.uniform(*self.TRAIN_GUST_FREQ))
            turb_std = float(task_rng.uniform(0.01, self.TRAIN_TURB_STD[1]))

            # 50% chance of impulse
            impulse_mag = float(task_rng.uniform(*self.TRAIN_IMPULSE_MAG)) if task_rng.random() > 0.5 else 0.0
            impulse_dur = 0.10
            impulse_onset = round(float(task_rng.uniform(1.0, 4.0)), 2)

            noise_scale = float(task_rng.uniform(*self.TRAIN_NOISE_SCALE))
            # 50% chance of mild motor degradation within training range
            motor_deg = float(task_rng.uniform(*self.TRAIN_MOTOR_DEG)) if task_rng.random() > 0.5 else 0.0
            if motor_deg > 0.0:
                degraded_motors = [int(m) for m in task_rng.choice([0, 1, 2, 3], size=2, replace=False)]
            else:
                degraded_motors = []

            mass_scale = round(float(task_rng.uniform(*self.TRAIN_MASS_SCALE)), 3)
            inertia_scale = round(float(task_rng.uniform(*self.TRAIN_INERTIA_SCALE)), 3)
            waypoint_seed = int(self.master_seed + 20000 + i)

            flags = {
                "has_wind": wind_mag > 0.5,
                "has_gusts": gust_mag > 0.0,
                "has_impulse": impulse_mag > 0.0,
                "has_sensor_noise": noise_scale > 1.2,
                "has_motor_degradation": motor_deg > 0.0,
                "is_compound": sum([wind_mag > 0.5, impulse_mag > 0.0, noise_scale > 1.2, motor_deg > 0.0]) > 1
            }

            task = MetaTask(
                task_id=f"val_task_{i+1:03d}",
                seed=task_seed,
                split=TaskSplit.VALIDATION.value,
                wind_magnitude=round(wind_mag, 3),
                wind_direction=wind_dir,
                gust_magnitude=round(gust_mag, 3),
                gust_frequency=round(gust_freq, 3),
                turbulence_std=round(turb_std, 3),
                impulse_magnitude=round(impulse_mag, 3),
                impulse_duration=round(impulse_dur, 2),
                impulse_onset=impulse_onset,
                sensor_noise_scale=round(noise_scale, 3),
                motor_degradation=round(motor_deg, 3),
                degraded_motors=degraded_motors,
                mass_scale=mass_scale,
                inertia_scale=inertia_scale,
                waypoint_seed=waypoint_seed,
                compound_disturbance_flags=flags
            )
            tasks.append(task)

        return tasks

    def generate_ood_test_tasks(self) -> List[MetaTask]:
        """Generate exactly 30 Out-of-Distribution (OOD) test tasks."""
        rng = np.random.Generator(np.random.PCG64(self.master_seed + 3000))
        tasks: List[MetaTask] = []

        # Mixture for OOD tasks:
        # 6 Severe Wind & Gusts ([3.0, 5.5] m/s)
        # 6 High Impulse Shocks ([4.0, 6.0] N)
        # 6 Extreme Sensor Noise ([3.5x, 5.0x])
        # 6 Severe Motor Loss ([30%, 70%] LoE on asymmetric rotors)
        # 6 Multi-Modal Compound OOD (Wind + Sensor + Motor Degradation)
        ood_categories = ["extreme_wind", "extreme_impulse", "extreme_noise", "severe_motor_deg", "compound_ood"]

        for i in range(self.OOD_COUNT):
            task_seed = int(self.master_seed + 3000 + i)
            task_rng = np.random.Generator(np.random.PCG64(task_seed))
            category = ood_categories[i % len(ood_categories)]

            # Default to nominal baseline
            wind_mag = 0.0
            gust_mag = 0.0
            gust_freq = 1.0
            turb_std = 0.0
            impulse_mag = 0.0
            impulse_dur = 0.10
            impulse_onset = round(float(task_rng.uniform(1.0, 3.5)), 2)
            noise_scale = 1.0
            motor_deg = 0.0
            degraded_motors: List[int] = []

            # Populate category specific OOD ranges
            if category == "extreme_wind":
                wind_mag = float(task_rng.uniform(*self.OOD_WIND_MAG))
                gust_mag = float(task_rng.uniform(*self.OOD_GUST_MAG))
                gust_freq = float(task_rng.uniform(*self.OOD_GUST_FREQ))
                turb_std = float(task_rng.uniform(*self.OOD_TURB_STD))
            elif category == "extreme_impulse":
                impulse_mag = float(task_rng.uniform(*self.OOD_IMPULSE_MAG))
                wind_mag = float(task_rng.uniform(1.0, 2.0))
            elif category == "extreme_noise":
                noise_scale = float(task_rng.uniform(*self.OOD_NOISE_SCALE))
                wind_mag = float(task_rng.uniform(1.0, 2.0))
            elif category == "severe_motor_deg":
                # Strict separation: motor degradation is in [0.30, 0.70]
                motor_deg = float(task_rng.uniform(*self.OOD_MOTOR_DEG))
                # Asymmetric motor degradation on rotors 0 and 1 or 1 and 2
                degraded_motors = [0, 1] if task_rng.random() > 0.5 else [1, 2]
            elif category == "compound_ood":
                wind_mag = float(task_rng.uniform(3.0, 4.5))
                gust_mag = float(task_rng.uniform(1.5, 3.0))
                gust_freq = float(task_rng.uniform(2.0, 4.0))
                turb_std = float(task_rng.uniform(0.15, 0.30))
                noise_scale = float(task_rng.uniform(3.0, 4.5))
                motor_deg = float(task_rng.uniform(0.30, 0.55))
                degraded_motors = [0, 1]
                impulse_mag = float(task_rng.uniform(3.5, 5.0)) if task_rng.random() > 0.5 else 0.0

            wind_dir = self._sample_unit_vector_3d(task_rng)
            mass_scale = round(float(task_rng.uniform(*self.OOD_MASS_SCALE)), 3)
            inertia_scale = round(float(task_rng.uniform(*self.OOD_INERTIA_SCALE)), 3)
            waypoint_seed = int(self.master_seed + 30000 + i)

            flags = {
                "has_wind": wind_mag > 0.5,
                "has_gusts": gust_mag > 0.0,
                "has_impulse": impulse_mag > 0.0,
                "has_sensor_noise": noise_scale > 2.0,
                "has_motor_degradation": motor_deg >= 0.30,
                "is_compound": sum([wind_mag > 0.5, impulse_mag > 0.0, noise_scale > 2.0, motor_deg >= 0.30]) > 1
            }

            task = MetaTask(
                task_id=f"ood_test_task_{i+1:03d}",
                seed=task_seed,
                split=TaskSplit.OOD_TEST.value,
                wind_magnitude=round(wind_mag, 3),
                wind_direction=wind_dir,
                gust_magnitude=round(gust_mag, 3),
                gust_frequency=round(gust_freq, 3),
                turbulence_std=round(turb_std, 3),
                impulse_magnitude=round(impulse_mag, 3),
                impulse_duration=round(impulse_dur, 2),
                impulse_onset=impulse_onset,
                sensor_noise_scale=round(noise_scale, 3),
                motor_degradation=round(motor_deg, 3),
                degraded_motors=degraded_motors,
                mass_scale=mass_scale,
                inertia_scale=inertia_scale,
                waypoint_seed=waypoint_seed,
                compound_disturbance_flags=flags
            )
            tasks.append(task)

        return tasks

    def generate_all_tasks(self) -> Dict[str, List[MetaTask]]:
        """Generate the full dictionary of 130 tasks across train, val, and ood_test splits."""
        train_tasks = self.generate_train_tasks()
        val_tasks = self.generate_validation_tasks()
        ood_tasks = self.generate_ood_test_tasks()

        manifest = {
            TaskSplit.TRAIN.value: train_tasks,
            TaskSplit.VALIDATION.value: val_tasks,
            TaskSplit.OOD_TEST.value: ood_tasks,
        }
        self.validate_task_manifest(manifest)
        return manifest

    def validate_task_manifest(self, manifest: Dict[str, List[MetaTask]]) -> None:
        """Validate strict integrity, count, uniqueness, range boundaries, and OOD separation.
        
        Raises
        ------
        ValueError
            If any integrity, range, or leakage violation is detected.
        """
        train = manifest.get(TaskSplit.TRAIN.value, [])
        val = manifest.get(TaskSplit.VALIDATION.value, [])
        ood = manifest.get(TaskSplit.OOD_TEST.value, [])

        # 1. Check exact counts
        if len(train) != self.TRAIN_COUNT:
            raise ValueError(f"Train task count mismatch: expected {self.TRAIN_COUNT}, got {len(train)}")
        if len(val) != self.VAL_COUNT:
            raise ValueError(f"Validation task count mismatch: expected {self.VAL_COUNT}, got {len(val)}")
        if len(ood) != self.OOD_COUNT:
            raise ValueError(f"OOD test task count mismatch: expected {self.OOD_COUNT}, got {len(ood)}")

        total_tasks = train + val + ood
        if len(total_tasks) != self.TOTAL_COUNT:
            raise ValueError(f"Total task count mismatch: expected {self.TOTAL_COUNT}, got {len(total_tasks)}")

        # 2. Check uniqueness of Task IDs
        all_ids = [t.task_id for t in total_tasks]
        if len(set(all_ids)) != len(all_ids):
            duplicates = [tid for tid in all_ids if all_ids.count(tid) > 1]
            raise ValueError(f"Duplicate task IDs found: {set(duplicates)}")

        # 3. Check disjointness of Seed Namespaces
        train_seeds = set(t.seed for t in train)
        val_seeds = set(t.seed for t in val)
        ood_seeds = set(t.seed for t in ood)

        if train_seeds.intersection(val_seeds):
            raise ValueError(f"Train and Val seeds overlap: {train_seeds.intersection(val_seeds)}")
        if train_seeds.intersection(ood_seeds):
            raise ValueError(f"Train and OOD seeds overlap: {train_seeds.intersection(ood_seeds)}")
        if val_seeds.intersection(ood_seeds):
            raise ValueError(f"Val and OOD seeds overlap: {val_seeds.intersection(ood_seeds)}")

        # 4. Range Validation & Leakage Prevention
        for t in train:
            if not (self.TRAIN_WIND_MAG[0] <= t.wind_magnitude <= self.TRAIN_WIND_MAG[1] + 1e-5):
                raise ValueError(f"Train wind magnitude out of bounds in {t.task_id}: {t.wind_magnitude}")
            if not (self.TRAIN_MOTOR_DEG[0] <= t.motor_degradation <= self.TRAIN_MOTOR_DEG[1] + 1e-5):
                raise ValueError(f"Train motor degradation out of bounds in {t.task_id}: {t.motor_degradation}")
            if not (self.TRAIN_NOISE_SCALE[0] <= t.sensor_noise_scale <= self.TRAIN_NOISE_SCALE[1] + 1e-5):
                raise ValueError(f"Train sensor noise out of bounds in {t.task_id}: {t.sensor_noise_scale}")

        for t in val:
            if not (self.TRAIN_WIND_MAG[0] <= t.wind_magnitude <= self.TRAIN_WIND_MAG[1] + 1e-5):
                raise ValueError(f"Val wind magnitude out of bounds in {t.task_id}: {t.wind_magnitude}")
            if not (self.TRAIN_MOTOR_DEG[0] <= t.motor_degradation <= self.TRAIN_MOTOR_DEG[1] + 1e-5):
                raise ValueError(f"Val motor degradation out of bounds in {t.task_id}: {t.motor_degradation}")

        for t in ood:
            # For OOD tasks with motor degradation, must be strictly >= 0.30
            if t.motor_degradation > 0.0 and t.motor_degradation < self.OOD_MOTOR_DEG[0] - 1e-5:
                raise ValueError(f"OOD motor degradation leaked into training range in {t.task_id}: {t.motor_degradation}")
            if t.motor_degradation > self.OOD_MOTOR_DEG[1] + 1e-5:
                raise ValueError(f"OOD motor degradation exceeds maximum in {t.task_id}: {t.motor_degradation}")

    def save_manifest(self, filepath: str) -> None:
        """Generate and save complete task manifest to a JSON file."""
        manifest = self.generate_all_tasks()
        serializable = {
            split: [t.to_dict() for t in tasks]
            for split, tasks in manifest.items()
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"[TaskGenerator] Saved {self.TOTAL_COUNT} tasks to: {filepath}")

    @classmethod
    def load_manifest(cls, filepath: str) -> Dict[str, List[MetaTask]]:
        """Load and deserialize task manifest from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = {
            split: [MetaTask.from_dict(t) for t in tasks]
            for split, tasks in data.items()
        }
        # Run validation on loaded manifest
        dummy_gen = cls()
        dummy_gen.validate_task_manifest(manifest)
        return manifest
