# MCR-UAV: Reconfiguration Trajectory Buffer & Multi-Task Storage

**Document Version:** 1.0.0  
**Project:** Meta-Contextual Reconfiguration of Hierarchical UAV Control (MCR-UAV)  
**Date:** 2026-08-12  
**Implementation Source:** `models/trajectory_buffer.py`  
**Unit Tests:** `tests/test_trajectory_buffer.py`  

---

## 1. Motivation

Meta-Reinforcement Learning (Meta-RL) algorithms require temporal history transitions to infer latent environmental dynamics ($z_t = f_\phi(H_t)$) and evaluate multi-tier reconfiguration policies ($c_t = g_\theta(z_t)$). To support multi-task meta-training, meta-validation, and out-of-distribution (OOD) testing without data leakage or temporal corruption, MCR-UAV implements a dedicated **Reconfiguration Trajectory Buffer**.

This data infrastructure enforces:
1. Strict numerical integrity and finite-state assertions.
2. Causal rolling history window extraction ($L = 20$, $d_h = 52$).
3. Complete task and split isolation (preventing cross-task transition mixing).
4. Full compatibility with the Transformer Context Encoder and Meta-Supervisor.

---

## 2. Transition Schema and Dimensionality

Each transition step is encapsulated in a validated `Transition` object:

$$x_t = \begin{bmatrix} s_t \\ a_t \\ r_t \\ \Delta s_t \end{bmatrix} \in \mathbb{R}^{52}$$

### Detailed Transition Fields:

| Field | Type | Dimension | Description |
| :--- | :--- | :---: | :--- |
| **$s_t$** | `np.ndarray` (float32) | **$24$** | Full state observation: $p_t (3), v_t (3), q_t (3), \omega_t (3), e_t (3), d_t (1), u_t (5), a_{t-1} (3)$ |
| **$a_t$** | `np.ndarray` (float32) | **$3$** | High-level velocity command: $[v_x^{\text{cmd}}, v_y^{\text{cmd}}, v_z^{\text{cmd}}]$ |
| **$r_t$** | `float` (float32) | **$1$** | Immediate scalar reward signal |
| **$s_{t+1}$** | `np.ndarray` (float32) | **$24$** | Next state observation |
| **$\Delta s_t$** | `np.ndarray` (float32) | **$24$** | State increment ($\Delta s_t = s_{t+1} - s_t$) representing acceleration / dynamics |
| **`done`** | `bool` | **$1$** | Episode termination / truncation flag |
| **`task_id`** | `str` | $-$ | Unique identifier of the originating task |
| **`episode_id`** | `int` | $-$ | Episode index within the task |
| **$z_t$** *(optional)* | `np.ndarray` (float32) | **$16$** | Latent context vector inferred by Transformer Context Encoder |
| **$\lambda_{\text{RL}}$** *(optional)* | `float` | **$1$** | RL blending authority factor $\in [0.0, 1.0]$ |
| **$\alpha_Q, \alpha_R$** *(optional)* | `float` | **$2$** | MPC state and input cost scaling factors $\in [0.2, 5.0]$ |
| **$\alpha_P, \alpha_I, \alpha_D$** *(optional)* | `float` | **$3$** | PID gain scaling factors $\in [0.5, 2.0], [0.2, 2.5], [0.5, 2.0]$ |
| **$H$** *(optional)* | `int` | **$1$** | Discrete MPC prediction horizon $\in \{10, 20, 30\}$ |

---

## 3. Rolling History Window and Causal Padding Mechanism

For any timestep $t \ge 0$ in an episode, `EpisodeBuffer.get_history(t, window_size=20)` constructs the chronological history tensor $H_t \in \mathbb{R}^{20 \times 52}$:

### 3.1. Short History Phase ($t < 20$):
When fewer than $20$ transitions have occurred ($k = t + 1 \in \{1, \dots, 19\}$):
1. Transitions from index $0$ to $t$ are placed in slots $[0, \dots, k-1]$.
2. Padded zero vectors fill remaining slots $[k, \dots, 19]$.
3. A boolean `padding_mask` of shape $(20,)$ is generated:
   $$\text{padding\_mask}[i] = \begin{cases} \text{False}, & 0 \le i < k \\ \text{True}, & k \le i < 20 \end{cases}$$
4. `valid_len = k` is returned.
5. The final valid transition at index $k-1$ exactly corresponds to timestep $t$.

### 3.2. Steady-State Phase ($t \ge 19$):
1. Extracts exact rolling slice $[t-19, \dots, t]$.
2. `padding_mask` is all `False`.
3. `valid_len = 20`.

```
Short History (t = 4, k = 5):
┌────┬────┬────┬────┬────┬────┬────┬────┬─────┬────┐
│x_0 │x_1 │x_2 │x_3 │x_4 │ 0  │ 0  │ 0  │ ... │ 0  │  Shape: [20, 52]
└────┴────┴────┴────┴────┴────┴────┴────┴─────┴────┘
 ◄── Real Transitions ──► ◄────── Padded Slots ─────►
 [F]  [F]  [F]  [F]  [F]  [T]  [T]  [T]   ...   [T]   Mask: [20] (True = ignore)
```

---

## 4. Task and Dataset Split Isolation Architecture

Data integrity across Meta-RL splits is strictly enforced via two abstraction tiers:

```
                            ┌────────────────────────────────────────┐
                            │          TaskTrajectoryStore           │
                            └───────────────────┬────────────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
  ┌─────────────────────────────┐┌─────────────────────────────┐┌─────────────────────────────┐
  │      TRAIN SPLIT STORE      ││       VAL SPLIT STORE       ││     OOD_TEST SPLIT STORE    │
  │  task_001, ..., task_080    ││  val_001, ..., val_020     ││  ood_001, ..., ood_030     │
  └──────────────┬──────────────┘└──────────────┬──────────────┘└──────────────┬──────────────┘
                 │                              │                              │
                 ▼                              ▼                              ▼
      ┌────────────────────┐         ┌────────────────────┐         ┌────────────────────┐
      │   EpisodeBuffer    │         │   EpisodeBuffer    │         │   EpisodeBuffer    │
      │ • task_id          │         │ • task_id          │         │ • task_id          │
      │ • task_split       │         │ • task_split       │         │ • task_split       │
      │ • episode_id       │         │ • episode_id       │         │ • episode_id       │
      │ • is_finalized     │         │ • is_finalized     │         │ • is_finalized     │
      │ • transitions: []  │         │ • transitions: []  │         │ • transitions: []  │
      └────────────────────┘         └────────────────────┘         └────────────────────┘
```

### Non-Negotiable Isolation Invariants:
1. **Task ID Validation:** Appending a transition with `task_id="task_B"` into an `EpisodeBuffer` initialized for `"task_A"` immediately throws a `ValueError`.
2. **Split Consistency:** Registering an episode with split `"VAL"` under a task previously designated `"TRAIN"` raises an exception.
3. **Deterministic Retrieval:** `store.get_task(task_id)` and `store.get_split(split)` return episodes and transitions in strictly deterministic chronological order.

---

## 5. Numerical Integrity Assertions

The buffer performs active sanitization on every numerical array:
* Rejects any `np.nan` in state, action, reward, next state, delta state, or latent context.
* Rejects any `np.inf` or `-np.inf`.
* Asserts discrete horizon $H \in \{10, 20, 30\}$ when provided.
* Asserts scalar reward $r_t$ is a finite floating-point value.

---

## 6. Serialization and Persistence

* **Method:** `TaskTrajectoryStore.save(filepath)` and `TaskTrajectoryStore.load(filepath)`.
* **Format:** Deterministic PyTorch binary serialization (`.pt`).
* **Cross-Platform Safety:** Fully compatible with Windows, Linux, and macOS without file-lock collisions.
* **Verification:** Serialized and reloaded buffers produce exact bitwise identical task structures, metadata, and history tensors.

---

## 7. Capacity Management

* Configurable bounds:
  - `max_episodes_per_task` (default: $100$)
  - `max_tasks` (default: $500$)
  - `max_transitions` per episode (default: $1000$)
* **Overflow Behavior:** Raises `BufferError` on capacity violation unless `overwrite=True` is explicitly configured. When `overwrite=True`, oldest episodes are systematically evicted (FIFO).

---

## 8. Unit Test Suite Summary (`tests/test_trajectory_buffer.py`)

All $23$ unit test cases passed with $100\%$ success rate:

| Test Case | Property Verified | Status |
| :--- | :--- | :---: |
| `test_transition_creation_and_auto_delta` | Auto $\Delta s_t = s_{t+1} - s_t$ and $52$-dim context vector | **PASSED** |
| `test_transition_dimension_validation` | Strict rejection of invalid shapes ($s_t \ne 24$, $a_t \ne 3$) | **PASSED** |
| `test_transition_nan_rejection` | ValueError thrown on NaN input array | **PASSED** |
| `test_transition_inf_rejection` | ValueError thrown on Inf input array | **PASSED** |
| `test_episode_append_and_ordering` | Sequential appending preserves temporal ordering | **PASSED** |
| `test_episode_finalization_immutability` | Finalized episodes strictly reject new transitions | **PASSED** |
| `test_rolling_history_extraction[0..25]` | Exact history shapes $(20, 52)$, padding masks, and valid lens | **PASSED** |
| `test_transformer_context_encoder_compat` | Direct forward-pass compatibility with $f_\phi$ model | **PASSED** |
| `test_task_isolation_in_episode_buffer` | Rejection of cross-task transition contamination | **PASSED** |
| `test_task_trajectory_store_isolation_splits` | Partitioning across TRAIN, VAL, and OOD_TEST splits | **PASSED** |
| `test_multiple_episodes_per_task` | Multi-episode indexing under single task key | **PASSED** |
| `test_capacity_overflow_error_and_overwrite` | BufferError on overflow / FIFO eviction on overwrite | **PASSED** |
| `test_latent_and_reconfiguration_storage` | Correct logging of $z_t$ and $c_t$ multi-tier parameters | **PASSED** |
| `test_serialization_roundtrip` | Exact binary save/load preservation | **PASSED** |
| `test_reward_non_finite_rejection` | Rejection of NaN / Inf reward scalars | **PASSED** |
| `test_invalid_split_string_rejection` | Rejection of unapproved split names | **PASSED** |
| `test_horizon_validation_in_transition` | Assertion of $H \in \{10, 20, 30\}$ | **PASSED** |
| `test_empty_and_out_of_range_history_errors` | Proper exception handling on empty/out-of-bounds queries | **PASSED** |
| `test_deterministic_retrieval_consistency` | Exact ordering across repeated read queries | **PASSED** |

---

## 9. Implementation Status Summary

* **Currently Implemented (Phase 8):**
  - Standalone `models/trajectory_buffer.py` module.
  - `Transition`, `EpisodeBuffer`, `TaskTrajectoryStore`, and `TaskSplitEnum`.
  - Comprehensive unit test suite ($23/23$ tests passing).
  - Total full-workspace unit tests: $74/74$ passing.
* **Planned for Future Phases:**
  - Phase 9: Hierarchical Environment Reconfiguration Integration ($c_t \to \text{RL/MPC/PID}$).
  - Phase 10: Multi-Task Meta-RL Loss Formulation.
  - Phase 15: Meta-RL Training Loop with Trajectory Buffer sampling.
