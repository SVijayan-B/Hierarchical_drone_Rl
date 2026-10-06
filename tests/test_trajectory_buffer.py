"""Comprehensive Unit Tests for MCR-UAV Reconfiguration Trajectory Buffer."""

import os
import sys
import tempfile
import pytest
import torch
import numpy as np

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.trajectory_buffer import (
    EpisodeBuffer,
    TaskSplitEnum,
    TaskTrajectoryStore,
    Transition,
)
from models.transformer_context_encoder import TransformerContextEncoder


def _create_sample_transition(
    task_id: str = "train_task_001",
    episode_id: int = 1,
    step_idx: int = 0,
    with_optional: bool = False,
) -> Transition:
    """Helper to create a valid deterministic transition."""
    s_t = np.full(24, step_idx * 0.1, dtype=np.float32)
    a_t = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    r_t = -0.5
    s_next = np.full(24, (step_idx + 1) * 0.1, dtype=np.float32)
    done = False
    
    if with_optional:
        z_t = np.full(16, 0.05, dtype=np.float32)
        return Transition(
            s_t=s_t,
            a_t=a_t,
            r_t=r_t,
            s_next=s_next,
            done=done,
            task_id=task_id,
            episode_id=episode_id,
            z_t=z_t,
            lambda_rl=0.8,
            alpha_q=1.2,
            alpha_r=0.9,
            alpha_p=1.1,
            alpha_i=1.0,
            alpha_d=1.05,
            horizon=20,
        )
    
    return Transition(
        s_t=s_t,
        a_t=a_t,
        r_t=r_t,
        s_next=s_next,
        done=done,
        task_id=task_id,
        episode_id=episode_id,
    )


def test_transition_creation_and_auto_delta():
    """Test valid transition creation and auto computation of delta_s_t."""
    trans = _create_sample_transition()
    assert trans.s_t.shape == (24,)
    assert trans.a_t.shape == (3,)
    assert trans.s_next.shape == (24,)
    assert trans.delta_s_t.shape == (24,)
    assert np.allclose(trans.delta_s_t, trans.s_next - trans.s_t)
    
    x_t = trans.get_context_vector()
    assert x_t.shape == (52,)
    assert np.allclose(x_t[:24], trans.s_t)
    assert np.allclose(x_t[24:27], trans.a_t)
    assert x_t[27] == trans.r_t
    assert np.allclose(x_t[28:], trans.delta_s_t)


def test_transition_dimension_validation():
    """Test rejection of malformed tensor shapes."""
    # Bad state dimension
    with pytest.raises(ValueError, match="Expected s_t shape"):
        Transition(
            s_t=np.zeros(20),
            a_t=np.zeros(3),
            r_t=0.0,
            s_next=np.zeros(24),
            done=False,
            task_id="t1",
            episode_id=1,
        )
        
    # Bad action dimension
    with pytest.raises(ValueError, match="Expected a_t shape"):
        Transition(
            s_t=np.zeros(24),
            a_t=np.zeros(4),
            r_t=0.0,
            s_next=np.zeros(24),
            done=False,
            task_id="t1",
            episode_id=1,
        )


def test_transition_nan_rejection():
    """Test strict rejection of NaN values."""
    s_nan = np.zeros(24)
    s_nan[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        Transition(
            s_t=s_nan,
            a_t=np.zeros(3),
            r_t=0.0,
            s_next=np.zeros(24),
            done=False,
            task_id="t1",
            episode_id=1,
        )


def test_transition_inf_rejection():
    """Test strict rejection of Inf values."""
    a_inf = np.zeros(3)
    a_inf[1] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        Transition(
            s_t=np.zeros(24),
            a_t=a_inf,
            r_t=0.0,
            s_next=np.zeros(24),
            done=False,
            task_id="t1",
            episode_id=1,
        )


def test_episode_append_and_ordering():
    """Test sequential transition appending and temporal preservation."""
    ep = EpisodeBuffer(task_id="train_task_001", task_split="TRAIN", episode_id=1)
    
    for i in range(15):
        ep.append_transition(_create_sample_transition(step_idx=i))
        
    assert ep.length() == 15
    transitions = ep.get_transitions()
    for idx, t in enumerate(transitions):
        assert np.isclose(t.s_t[0], idx * 0.1)


def test_episode_finalization_immutability():
    """Test that finalized episodes reject new transitions."""
    ep = EpisodeBuffer(task_id="train_task_001", task_split="TRAIN", episode_id=1)
    ep.append_transition(_create_sample_transition(step_idx=0))
    ep.finalize_episode()
    assert ep.is_finalized
    
    with pytest.raises(RuntimeError, match="Cannot append transitions to a finalized episode"):
        ep.append_transition(_create_sample_transition(step_idx=1))


@pytest.mark.parametrize("target_step,expected_valid_len", [
    (0, 1),
    (4, 5),
    (9, 10),
    (19, 20),
    (25, 20),
])
def test_rolling_history_extraction_and_padding_mask(target_step, expected_valid_len):
    """Test rolling history window extraction for varying timesteps."""
    ep = EpisodeBuffer(task_id="train_task_001", task_split="TRAIN", episode_id=1)
    for i in range(30):
        ep.append_transition(_create_sample_transition(step_idx=i))
        
    hist, pad_mask, valid_len = ep.get_history(t=target_step, window_size=20, as_torch=True)
    
    assert hist.shape == (20, 52)
    assert pad_mask.shape == (20,)
    assert valid_len == expected_valid_len
    
    # Verify padding mask
    if expected_valid_len < 20:
        assert (pad_mask[:expected_valid_len] == False).all()
        assert (pad_mask[expected_valid_len:] == True).all()
    else:
        assert (pad_mask == False).all()
        
    # Verify final valid token matches timestep target_step
    final_valid_idx = valid_len - 1
    final_context = hist[final_valid_idx].numpy()
    expected_s_t = np.full(24, target_step * 0.1, dtype=np.float32)
    assert np.allclose(final_context[:24], expected_s_t)


def test_transformer_context_encoder_compatibility():
    """Test direct compatibility of extracted history with TransformerContextEncoder."""
    model = TransformerContextEncoder()
    model.eval()
    
    ep = EpisodeBuffer(task_id="train_task_001", task_split="TRAIN", episode_id=1)
    for i in range(25):
        ep.append_transition(_create_sample_transition(step_idx=i))
        
    for t in [0, 4, 9, 19, 24]:
        hist, pad_mask, valid_len = ep.get_history(t=t, window_size=20, as_torch=True)
        # Add batch dimension: [1, 20, 52], [1, 20]
        hist_b = hist.unsqueeze(0)
        pad_mask_b = pad_mask.unsqueeze(0)
        valid_lens_b = torch.tensor([valid_len], dtype=torch.long)
        
        with torch.no_grad():
            z = model(hist_b, padding_mask=pad_mask_b, valid_lens=valid_lens_b)
            
        assert z.shape == (1, 16)
        assert torch.isfinite(z).all()


def test_task_isolation_in_episode_buffer():
    """Test that EpisodeBuffer strictly rejects transitions from different tasks."""
    ep = EpisodeBuffer(task_id="train_task_001", task_split="TRAIN", episode_id=1)
    
    # Valid append
    ep.append_transition(_create_sample_transition(task_id="train_task_001"))
    
    # Invalid append from different task
    with pytest.raises(ValueError, match="Task ID mismatch"):
        ep.append_transition(_create_sample_transition(task_id="train_task_002"))


def test_task_trajectory_store_isolation_and_splits():
    """Test TaskTrajectoryStore partition across TRAIN, VAL, and OOD_TEST."""
    store = TaskTrajectoryStore()
    
    # Add train episodes
    for t_id in ["train_001", "train_002"]:
        ep = EpisodeBuffer(task_id=t_id, task_split="TRAIN", episode_id=1)
        ep.append_transition(_create_sample_transition(task_id=t_id))
        store.add_episode(t_id, ep)
        
    # Add val episode
    ep_val = EpisodeBuffer(task_id="val_001", task_split="VAL", episode_id=1)
    ep_val.append_transition(_create_sample_transition(task_id="val_001"))
    store.add_episode("val_001", ep_val)
    
    # Add ood episode
    ep_ood = EpisodeBuffer(task_id="ood_001", task_split="OOD_TEST", episode_id=1)
    ep_ood.append_transition(_create_sample_transition(task_id="ood_001"))
    store.add_episode("ood_001", ep_ood)
    
    assert store.num_tasks() == 4
    assert store.num_tasks("TRAIN") == 2
    assert store.num_tasks("VAL") == 1
    assert store.num_tasks("OOD_TEST") == 1
    assert store.num_episodes() == 4
    
    # Check retrieving task A never returns task B
    train_001_eps = store.get_task("train_001")
    assert len(train_001_eps) == 1
    assert train_001_eps[0].task_id == "train_001"
    
    train_split_tasks = store.get_split("TRAIN")
    assert set(train_split_tasks.keys()) == {"train_001", "train_002"}
    assert "val_001" not in train_split_tasks
    assert "ood_001" not in train_split_tasks


def test_multiple_episodes_per_task():
    """Test storing multiple episodes under the same task key."""
    store = TaskTrajectoryStore(max_episodes_per_task=10)
    
    for ep_id in range(1, 6):
        ep = EpisodeBuffer(task_id="train_001", task_split="TRAIN", episode_id=ep_id)
        ep.append_transition(_create_sample_transition(task_id="train_001", episode_id=ep_id))
        store.add_episode("train_001", ep)
        
    assert len(store.get_task("train_001")) == 5
    assert store.num_episodes("TRAIN") == 5


def test_capacity_overflow_error_and_overwrite():
    """Test capacity enforcement and overwrite mode."""
    # Test strict buffer error
    strict_store = TaskTrajectoryStore(max_episodes_per_task=2, overwrite=False)
    for ep_id in range(1, 3):
        ep = EpisodeBuffer(task_id="t1", task_split="TRAIN", episode_id=ep_id)
        strict_store.add_episode("t1", ep)
        
    with pytest.raises(BufferError, match="Maximum episodes per task"):
        ep_extra = EpisodeBuffer(task_id="t1", task_split="TRAIN", episode_id=3)
        strict_store.add_episode("t1", ep_extra)
        
    # Test overwrite mode
    overwrite_store = TaskTrajectoryStore(max_episodes_per_task=2, overwrite=True)
    for ep_id in range(1, 4):
        ep = EpisodeBuffer(task_id="t1", task_split="TRAIN", episode_id=ep_id)
        overwrite_store.add_episode("t1", ep)
    assert len(overwrite_store.get_task("t1")) == 2
    # Oldest episode (1) was popped, 2 and 3 remain
    eps = overwrite_store.get_task("t1")
    assert eps[0].episode_id == 2
    assert eps[1].episode_id == 3


def test_latent_and_reconfiguration_storage():
    """Test storing and retrieving optional latent z_t and reconfiguration parameters."""
    ep = EpisodeBuffer(task_id="train_task_001", task_split="TRAIN", episode_id=1)
    trans = _create_sample_transition(with_optional=True)
    ep.append_transition(trans)
    
    retrieved = ep.get_transitions()[0]
    assert retrieved.z_t is not None
    assert retrieved.z_t.shape == (16,)
    assert retrieved.lambda_rl == 0.8
    assert retrieved.alpha_q == 1.2
    assert retrieved.alpha_r == 0.9
    assert retrieved.alpha_p == 1.1
    assert retrieved.alpha_i == 1.0
    assert retrieved.alpha_d == 1.05
    assert retrieved.horizon == 20


def test_serialization_roundtrip():
    """Test save and load roundtrip producing exact identical trajectory stores."""
    store = TaskTrajectoryStore()
    
    for t_id, split in [("train_001", "TRAIN"), ("val_001", "VAL"), ("ood_001", "OOD_TEST")]:
        ep = EpisodeBuffer(task_id=t_id, task_split=split, episode_id=1, seed=42)
        for step in range(5):
            ep.append_transition(_create_sample_transition(task_id=t_id, episode_id=1, step_idx=step, with_optional=True))
        store.add_episode(t_id, ep)
        
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "trajectory_store.pt")
        store.save(tmp_path)
        loaded_store = TaskTrajectoryStore.load(tmp_path)
        
        assert loaded_store.num_tasks() == store.num_tasks()
        assert loaded_store.num_episodes() == store.num_episodes()
        assert loaded_store.num_tasks("TRAIN") == 1
        assert loaded_store.num_tasks("VAL") == 1
        assert loaded_store.num_tasks("OOD_TEST") == 1
        
        orig_hist, orig_mask, orig_vlen = store.get_task("train_001")[0].get_history(3)
        load_hist, load_mask, load_vlen = loaded_store.get_task("train_001")[0].get_history(3)
        
        assert np.allclose(orig_hist, load_hist)
        assert np.array_equal(orig_mask, load_mask)
        assert orig_vlen == load_vlen


def test_reward_non_finite_rejection():
    """Test rejection of non-finite reward values."""
    with pytest.raises(ValueError, match="Reward r_t must be finite"):
        Transition(
            s_t=np.zeros(24),
            a_t=np.zeros(3),
            r_t=float("nan"),
            s_next=np.zeros(24),
            done=False,
            task_id="t1",
            episode_id=1,
        )


def test_invalid_split_string_rejection():
    """Test rejection of invalid task split strings."""
    with pytest.raises(ValueError, match="Invalid split name"):
        EpisodeBuffer(task_id="t1", task_split="UNKNOWN_SPLIT", episode_id=1)


def test_horizon_validation_in_transition():
    """Test that invalid horizon values are rejected."""
    with pytest.raises(ValueError, match="Horizon must be in"):
        Transition(
            s_t=np.zeros(24),
            a_t=np.zeros(3),
            r_t=0.0,
            s_next=np.zeros(24),
            done=False,
            task_id="t1",
            episode_id=1,
            horizon=17,  # invalid horizon
        )


def test_empty_episode_and_out_of_range_history_errors():
    """Test boundary errors when requesting history from empty or out of range episodes."""
    ep = EpisodeBuffer(task_id="t1", task_split="TRAIN", episode_id=1)
    with pytest.raises(ValueError, match="Cannot extract history from an empty episode buffer"):
        ep.get_history(0)
        
    ep.append_transition(_create_sample_transition(task_id="t1", step_idx=0))
    with pytest.raises(IndexError, match="out of range"):
        ep.get_history(5)
    with pytest.raises(IndexError, match="out of range"):
        ep.get_history(-1)


def test_deterministic_retrieval_consistency():
    """Verify that multiple successive retrievals preserve exact identical ordering."""
    store = TaskTrajectoryStore()
    for ep_id in range(5):
        ep = EpisodeBuffer(task_id="task_det", task_split="TRAIN", episode_id=ep_id)
        for s in range(4):
            ep.append_transition(_create_sample_transition(task_id="task_det", episode_id=ep_id, step_idx=s))
        store.add_episode("task_det", ep)
        
    retrieval1 = [t.s_t[0] for ep in store.get_task("task_det") for t in ep.get_transitions()]
    retrieval2 = [t.s_t[0] for ep in store.get_task("task_det") for t in ep.get_transitions()]
    assert retrieval1 == retrieval2

