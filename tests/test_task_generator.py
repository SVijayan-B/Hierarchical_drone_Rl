import json
import os
import sys
import tempfile
import pytest
import numpy as np

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.task_generator import MetaTask, TaskGenerator, TaskSplit


def test_deterministic_task_counts():
    """Test that TaskGenerator produces exact required counts: 80 train, 20 val, 30 ood."""
    gen = TaskGenerator(master_seed=42)
    manifest = gen.generate_all_tasks()
    
    assert len(manifest[TaskSplit.TRAIN.value]) == 80
    assert len(manifest[TaskSplit.VALIDATION.value]) == 20
    assert len(manifest[TaskSplit.OOD_TEST.value]) == 30
    assert len(manifest[TaskSplit.TRAIN.value]) + len(manifest[TaskSplit.VALIDATION.value]) + len(manifest[TaskSplit.OOD_TEST.value]) == 130


def test_unique_task_ids():
    """Test that all 130 task IDs are globally unique."""
    gen = TaskGenerator(master_seed=42)
    manifest = gen.generate_all_tasks()
    all_tasks = manifest["train"] + manifest["val"] + manifest["ood_test"]
    all_ids = [t.task_id for t in all_tasks]
    
    assert len(all_ids) == 130
    assert len(set(all_ids)) == 130


def test_disjoint_seed_namespaces():
    """Test that seeds across train, val, and ood_test splits are strictly disjoint."""
    gen = TaskGenerator(master_seed=42)
    manifest = gen.generate_all_tasks()
    
    train_seeds = set(t.seed for t in manifest["train"])
    val_seeds = set(t.seed for t in manifest["val"])
    ood_seeds = set(t.seed for t in manifest["ood_test"])
    
    assert len(train_seeds.intersection(val_seeds)) == 0
    assert len(train_seeds.intersection(ood_seeds)) == 0
    assert len(val_seeds.intersection(ood_seeds)) == 0


def test_training_and_validation_ranges():
    """Test that all training and validation task parameters fall strictly inside declared bounds."""
    gen = TaskGenerator(master_seed=42)
    manifest = gen.generate_all_tasks()
    
    for split in ["train", "val"]:
        for t in manifest[split]:
            assert 0.0 <= t.wind_magnitude <= 2.5 + 1e-4
            assert 0.0 <= t.gust_magnitude <= 1.5 + 1e-4
            assert 0.5 <= t.gust_frequency <= 2.0 + 1e-4
            assert 0.0 <= t.turbulence_std <= 0.15 + 1e-4
            if t.impulse_magnitude > 0.0:
                assert 1.0 <= t.impulse_magnitude <= 3.0 + 1e-4
            assert 1.0 <= t.sensor_noise_scale <= 2.0 + 1e-4
            assert 0.0 <= t.motor_degradation <= 0.25 + 1e-4
            assert 0.90 <= t.mass_scale <= 1.15 + 1e-4
            assert 0.90 <= t.inertia_scale <= 1.15 + 1e-4


def test_ood_separation_and_motor_degradation_boundary():
    """Test strict separation of OOD parameters (e.g. motor degradation >= 30%, wind >= 3.0 m/s)."""
    gen = TaskGenerator(master_seed=42)
    manifest = gen.generate_all_tasks()
    
    for t in manifest["ood_test"]:
        if t.motor_degradation > 0.0:
            # OOD motor degradation must be strictly >= 0.30 (30% LoE)
            assert t.motor_degradation >= 0.30 - 1e-4
            assert t.motor_degradation <= 0.70 + 1e-4
            assert len(t.degraded_motors) > 0
        
        if t.wind_magnitude > 2.5:
            assert t.wind_magnitude >= 3.0 - 1e-4
            assert t.wind_magnitude <= 5.5 + 1e-4

        if t.sensor_noise_scale > 2.0:
            assert t.sensor_noise_scale >= 3.5 - 1e-4
            assert t.sensor_noise_scale <= 5.0 + 1e-4

        if t.impulse_magnitude > 3.0:
            assert t.impulse_magnitude >= 4.0 - 1e-4
            assert t.impulse_magnitude <= 6.0 + 1e-4


def test_task_serialization_roundtrip():
    """Test dataclass to_dict, to_json, from_dict roundtrip fidelity."""
    gen = TaskGenerator(master_seed=42)
    tasks = gen.generate_train_tasks()
    original_task = tasks[0]
    
    # Dict roundtrip
    d = original_task.to_dict()
    reconstructed_from_dict = MetaTask.from_dict(d)
    assert reconstructed_from_dict == original_task
    
    # JSON roundtrip
    j_str = original_task.to_json()
    reconstructed_from_json = MetaTask.from_dict(json.loads(j_str))
    assert reconstructed_from_json == original_task


def test_master_seed_reproducibility():
    """Test that identical master seed produces 100% identical task lists."""
    gen1 = TaskGenerator(master_seed=42)
    gen2 = TaskGenerator(master_seed=42)
    
    manifest1 = gen1.generate_all_tasks()
    manifest2 = gen2.generate_all_tasks()
    
    for split in ["train", "val", "ood_test"]:
        for t1, t2 in zip(manifest1[split], manifest2[split]):
            assert t1.to_dict() == t2.to_dict()


def test_different_master_seed_variance():
    """Test that different master seeds produce different parameters."""
    gen1 = TaskGenerator(master_seed=42)
    gen2 = TaskGenerator(master_seed=99)
    
    manifest1 = gen1.generate_all_tasks()
    manifest2 = gen2.generate_all_tasks()
    
    # Check that at least some parameters vary across seeds
    diff_count = sum(
        t1.wind_magnitude != t2.wind_magnitude
        for t1, t2 in zip(manifest1["train"], manifest2["train"])
    )
    assert diff_count > 0


def test_save_and_load_manifest():
    """Test saving full manifest to disk and loading it back with validation."""
    gen = TaskGenerator(master_seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_manifest.json")
        gen.save_manifest(filepath)
        
        loaded = TaskGenerator.load_manifest(filepath)
        assert len(loaded["train"]) == 80
        assert len(loaded["val"]) == 20
        assert len(loaded["ood_test"]) == 30
        assert loaded["train"][0] == gen.generate_train_tasks()[0]
