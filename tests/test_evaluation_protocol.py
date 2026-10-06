import csv
import json
import os
import sys
import tempfile
import pytest

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.task_generator import MetaTask, TaskGenerator, TaskSplit
from evaluation.evaluation_protocol import EvaluationProtocol


def test_protocol_configuration_loading():
    """Test loading and validating configs/evaluation_protocol.yaml / json."""
    protocol = EvaluationProtocol(
        config_path="configs/evaluation_protocol.yaml",
        manifest_path="configs/meta_tasks_manifest.json",
        gui=False
    )
    cfg = protocol.cfg
        
    assert cfg["splits"]["train"]["task_count"] == 80
    assert cfg["splits"]["validation"]["task_count"] == 20
    assert cfg["splits"]["ood_test"]["task_count"] == 30
    assert "mlp" in cfg["baseline_checkpoints"]
    assert "transformer" in cfg["baseline_checkpoints"]


def test_manifest_validation_and_leakage_checks():
    """Test that EvaluationProtocol validates the 130-task manifest with zero leakage."""
    protocol = EvaluationProtocol(
        config_path="configs/evaluation_protocol.yaml",
        manifest_path="configs/meta_tasks_manifest.json",
        gui=False
    )
    assert len(protocol.manifest["train"]) == 80
    assert len(protocol.manifest["val"]) == 20
    assert len(protocol.manifest["ood_test"]) == 30


def test_mlp_and_transformer_baseline_loading():
    """Test loading both frozen baseline models and their respective VecNormalize wrappers."""
    protocol = EvaluationProtocol(
        config_path="configs/evaluation_protocol.yaml",
        manifest_path="configs/meta_tasks_manifest.json",
        gui=False
    )
    cfg = protocol.cfg
        
    mlp_model = cfg["baseline_checkpoints"]["mlp"]["model_path"]
    mlp_vecnorm = cfg["baseline_checkpoints"]["mlp"]["vecnorm_path"]
    trans_model = cfg["baseline_checkpoints"]["transformer"]["model_path"]
    trans_vecnorm = cfg["baseline_checkpoints"]["transformer"]["vecnorm_path"]
    
    assert os.path.exists(mlp_model)
    assert os.path.exists(mlp_vecnorm)
    assert os.path.exists(trans_model)
    assert os.path.exists(trans_vecnorm)


def test_single_task_evaluation_mlp_and_transformer():
    """Test single-task evaluation on MLP and Transformer models with short episode duration."""
    protocol = EvaluationProtocol(
        config_path="configs/evaluation_protocol.yaml",
        manifest_path="configs/meta_tasks_manifest.json",
        gui=False
    )
    sample_task = protocol.manifest["train"][0]
    
    # Test MLP evaluation
    mlp_records = protocol.evaluate_model(
        model_name="mlp",
        tasks=[sample_task],
        episodes_per_task=1,
        episode_sec=2.0
    )
    assert len(mlp_records) == 1
    assert mlp_records[0]["model"] == "mlp"
    assert mlp_records[0]["task_id"] == sample_task.task_id
    assert "rmse_tracking_error" in mlp_records[0]
    
    # Test Transformer evaluation
    trans_records = protocol.evaluate_model(
        model_name="transformer",
        tasks=[sample_task],
        episodes_per_task=1,
        episode_sec=2.0
    )
    assert len(trans_records) == 1
    assert trans_records[0]["model"] == "transformer"
    assert trans_records[0]["task_id"] == sample_task.task_id
    assert "rmse_tracking_error" in trans_records[0]


def test_summary_statistics_computation():
    """Test aggregated summary statistics computation and CSV schema."""
    protocol = EvaluationProtocol(
        config_path="configs/evaluation_protocol.yaml",
        manifest_path="configs/meta_tasks_manifest.json",
        gui=False
    )
    sample_tasks = protocol.manifest["train"][:2]
    records = protocol.evaluate_model(
        model_name="mlp",
        tasks=sample_tasks,
        episodes_per_task=1,
        episode_sec=1.5
    )
    summary = protocol.compute_summary_statistics(records)
    
    assert len(summary) == 1
    assert summary[0]["model"] == "mlp"
    assert summary[0]["split"] == "train"
    assert summary[0]["task_count"] == 2
    assert "mean_rmse" in summary[0]
    assert "mean_success_rate" in summary[0]
