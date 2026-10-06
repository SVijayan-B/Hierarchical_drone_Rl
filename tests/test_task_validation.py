"""Unit Tests for MCR-UAV Phase 4 Task Validation Module."""

import csv
import os
import sys
import tempfile
import pytest
import numpy as np

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.task_generator import MetaTask, TaskGenerator, TaskSplit
from evaluation.validate_tasks import TaskValidator


def test_manifest_and_task_loading():
    """Test loading MetaTask instances from the generated manifest."""
    manifest_path = "configs/meta_tasks_manifest.json"
    if not os.path.exists(manifest_path):
        gen = TaskGenerator(master_seed=42)
        gen.save_manifest(manifest_path)
        
    manifest = TaskGenerator.load_manifest(manifest_path)
    assert len(manifest["train"]) == 80
    assert len(manifest["val"]) == 20
    assert len(manifest["ood_test"]) == 30
    
    first_task = manifest["train"][0]
    assert isinstance(first_task, MetaTask)
    assert first_task.task_id == "train_task_001"


def test_task_validator_single_task_execution():
    """Test executing a short validation episode on a single train task."""
    gen = TaskGenerator(master_seed=42)
    train_tasks = gen.generate_train_tasks()
    task = train_tasks[0]
    
    validator = TaskValidator(gui=False, episode_sec=2.0)
    result = validator.validate_task(task)
    
    assert result["task_id"] == task.task_id
    assert result["simulation_valid"] == 1
    assert result["nan_detected"] == 0
    assert result["inf_detected"] == 0
    assert result["episode_steps"] > 0
    assert result["classification"] in ["VALID + SUCCESS", "VALID + BASELINE_FAILURE"]


def test_ood_task_motor_degradation_injection():
    """Test that OOD tasks with severe motor degradation execute with valid simulation state."""
    gen = TaskGenerator(master_seed=42)
    ood_tasks = gen.generate_ood_test_tasks()
    
    # Find a task with severe motor degradation
    deg_task = next(t for t in ood_tasks if t.motor_degradation >= 0.30)
    
    validator = TaskValidator(gui=False, episode_sec=2.0)
    result = validator.validate_task(deg_task)
    
    assert result["simulation_valid"] == 1
    assert result["nan_detected"] == 0
    assert result["inf_detected"] == 0
    # Check that motor degradation was recorded in result
    assert result["motor_degradation"] >= 0.30


def test_csv_schema_and_export():
    """Test CSV generation schema and header compatibility."""
    gen = TaskGenerator(master_seed=42)
    sample_tasks = gen.generate_train_tasks()[:2]
    
    validator = TaskValidator(gui=False, episode_sec=1.5)
    results = validator.validate_task_list(sample_tasks)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "test_task_val.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TaskValidator.CSV_HEADER)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
                
        # Read back and verify
        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert "simulation_valid" in rows[0]
            assert "baseline_failed" in rows[0]
            assert "nan_detected" in rows[0]
            assert "classification" in rows[0]
