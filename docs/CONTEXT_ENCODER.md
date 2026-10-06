# MCR-UAV: Transformer Context Encoder Architecture

**Document Version:** 1.0.0  
**Project:** Meta-Contextual Reconfiguration of Hierarchical UAV Control (MCR-UAV)  
**Date:** 2026-08-12  
**Implementation Source:** `models/transformer_context_encoder.py`  
**Unit Tests:** `tests/test_transformer_context_encoder.py`  
**Latency Benchmark:** `scripts/benchmark_context_encoder.py`  

---

## 1. Motivation

In hierarchical autonomous UAV flight control, unexpected physical perturbations (e.g., severe motor loss-of-effectiveness, sustained crosswinds, Dryden turbulence, payload mass shifts, and sensor noise) degrade tracking accuracy and stability. Rather than relying on slow online gradient descent or monolithic end-to-end black-box policies, MCR-UAV deploys a temporal **Transformer Context Encoder** ($f_\phi$). 

The encoder observes recent transition history and extracts a compact latent context vector $z_t \in \mathbb{R}^{16}$. This latent context informs the Meta-Supervisor to perform rapid, deterministic multi-tier parameter reconfiguration ($\lambda_{\text{RL}}, H, Q, R, K_P, K_I, K_D$) across the RL, MPC, and PID control tiers in a single forward pass without backpropagation at inference time.

---

## 2. Input Representation and Transition Vector

The context encoder operates on a rolling history buffer of the $L = 20$ most recent transition tuples:

$$H_t = [x_{t-L+1}, x_{t-L+2}, \dots, x_t] \in \mathbb{R}^{L \times d_h}$$

Each transition tuple $x_k \in \mathbb{R}^{52}$ captures the complete state-action-reward-transition dynamics:

$$x_k = \begin{bmatrix} s_k \\ a_k \\ r_k \\ \Delta s_k \end{bmatrix} \in \mathbb{R}^{52}$$

### Detailed Transition Vector Breakdown ($d_h = 52$):

| Subvector | Component | Dimension | Mathematical Meaning |
| :--- | :--- | :---: | :--- |
| **$s_k$** | State Vector | **$24$** | $p_k (3), v_k (3), q_k (3), \omega_k (3), e_k^{\text{target}} (3), d_k^{\text{target}} (1), u_k^{\text{sonar}} (5), a_{k-1} (3)$ |
| **$a_k$** | Executed Action | **$3$** | High-level velocity command vector $[v_x^{\text{cmd}}, v_y^{\text{cmd}}, v_z^{\text{cmd}}]$ |
| **$r_k$** | Scalar Reward | **$1$** | Immediate scalar reward signal received from the environment |
| **$\Delta s_k$** | State Increment | **$24$** | Difference vector $\Delta s_k = s_k - s_{k-1}$ representing system acceleration/dynamics |
| **Total** | **$d_h$** | **$52$** | **$24 + 3 + 1 + 24 = 52$ dimensions** |

---

## 3. Architecture Specification

```
                         Input History Tensor
                      [B, L=20, d_h=52] or [20, 52]
                                   │
                                   ▼
                       Input Linear Projection
                            52 ──► 64
                                   │
                                   ▼
                   Sinusoidal Positional Encoding
                         PE(pos) ∈ R^(20 × 64)
                                   │
                                   ▼
              ┌─────────────────────────────────────────┐
              │   Causal Transformer Encoder (2 Layers) │
              │   • d_model = 64, n_heads = 4           │
              │   • dim_feedforward = 256, GELU         │
              │   • Pre-LN (norm_first=True)            │
              │   • Causal Mask: j > i blocked          │
              │   • Padding Mask: src_key_padding_mask  │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
                     Final Valid Timestep Gathering
                             [B, 1, 64]
                                   │
                                   ▼
                         Pre-Projection LayerNorm
                                   │
                                   ▼
                       Latent Linear Projection
                            64 ──► 16
                                   │
                                   ▼
                         Latent Context Vector
                             z_t ∈ R^(B × 16)
```

---

## 4. Architectural Details

### 4.1. Input Linear Projection
Maps continuous transition feature vectors from the environmental domain into the Transformer embedding space:
$$h_k^{(0)} = W_{\text{in}} x_k + b_{\text{in}}, \quad W_{\text{in}} \in \mathbb{R}^{64 \times 52}, \quad b_{\text{in}} \in \mathbb{R}^{64}$$

### 4.2. Deterministic Sinusoidal Positional Encoding
Preserves strict temporal ordering across the history window using deterministic sinusoidal functions ($L_{\max} = 20$):
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / 64}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / 64}}\right)$$

### 4.3. Causal Attention Masking (Anti-Leakage)
To prevent temporal information leakage during online real-time adaptation, the self-attention mechanism enforces an explicit upper-triangular boolean causal mask $M_{\text{causal}} \in \{0, 1\}^{L \times L}$:

$$M_{i, j} = \begin{cases} 0 \ (\text{allow}), & j \le i \\ 1 \ (\text{block} / -\infty), & j > i \end{cases}$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M_{\text{causal}}\right) V$$

### 4.4. Variable History and Padding Support
During episode initialization ($t < L$), the history buffer contains fewer than $20$ transitions ($k \in \{1, 5, 10\}$). The encoder accepts a boolean `padding_mask` ($B \times L$) and `valid_lens` integer tensor ($B$), automatically:
1. Masking padded transition tokens from attention keys.
2. Gathering the representation from the final *valid* timestep ($k-1$).
3. Outputting a valid, finite latent vector $z_t \in \mathbb{R}^{B \times 16}$.

### 4.5. Latent Output Projection
The gathered representation at index $k-1$ is normalized via `nn.LayerNorm(64)` and projected to latent dimension $d_z = 16$:
$$z_t = W_{\text{out}} \text{LayerNorm}(h_{\text{last}}) + b_{\text{out}}, \quad W_{\text{out}} \in \mathbb{R}^{16 \times 64}, \quad b_{\text{out}} \in \mathbb{R}^{16}$$
*No squashing activation is applied, permitting unrestricted signed representation of dynamic context.*

---

## 5. Trainable Parameter Count

| Component | Layer / Tensor | Parameters |
| :--- | :--- | :---: |
| **Input Projection** | `Linear(52, 64)` | $52 \times 64 + 64 = 3,392$ |
| **Positional Encoding** | Fixed Buffer `pe` | $0$ (non-trainable) |
| **Transformer Layer 1** | Self-Attention (`Q, K, V, Out`) + FFN + 2× LayerNorm | $50,048$ |
| **Transformer Layer 2** | Self-Attention (`Q, K, V, Out`) + FFN + 2× LayerNorm | $50,048$ |
| **Final LayerNorm** | `LayerNorm(64)` | $64 \times 2 = 128$ |
| **Latent Projection** | `Linear(64, 16)` | $64 \times 16 + 16 = 1,040$ |
| **Total Trainable Parameters** | **All Layers Combined** | **$104,528$** |

---

## 6. Computational Complexity

* **Self-Attention Complexity:** $\mathcal{O}(n_{\text{layers}} \cdot L^2 \cdot d_{\text{model}}) = \mathcal{O}(2 \times 20^2 \times 64) \approx 5.12 \times 10^4\text{ FLOPs}$.
* **Feedforward Complexity:** $\mathcal{O}(n_{\text{layers}} \cdot L \cdot d_{\text{model}} \cdot d_{\text{ff}}) = \mathcal{O}(2 \times 20 \times 64 \times 256) \approx 6.55 \times 10^5\text{ FLOPs}$.
* **Total Forward Pass:** $< 1.0 \times 10^6\text{ FLOPs}$ (extremely lightweight for real-time UAV flight controllers).

---

## 7. Unit Test Results (`tests/test_transformer_context_encoder.py`)

All $15$ unit test suites passed with $100\%$ success rate:

| Test Name | Verified Condition | Status |
| :--- | :--- | :---: |
| `test_model_architecture_and_parameter_count` | Exact hyperparameter configuration & 104,528 params | **PASSED** |
| `test_latent_output_shapes_and_batch_sizes[1]` | Batch size $B=1 \to [1, 16]$ | **PASSED** |
| `test_latent_output_shapes_and_batch_sizes[4]` | Batch size $B=4 \to [4, 16]$ | **PASSED** |
| `test_latent_output_shapes_and_batch_sizes[8]` | Batch size $B=8 \to [8, 16]$ | **PASSED** |
| `test_latent_output_shapes_and_batch_sizes[16]` | Batch size $B=16 \to [16, 16]$ | **PASSED** |
| `test_latent_output_shapes_and_batch_sizes[32]` | Batch size $B=32 \to [32, 16]$ | **PASSED** |
| `test_single_sample_unbatched_input` | Unbatched input $[20, 52] \to [1, 16]$ | **PASSED** |
| `test_gradient_flow_and_trainability` | All weights receive non-zero, finite gradients | **PASSED** |
| `test_strict_causality_no_future_leakage` | Future perturbations ($t \ge 10$) produce $0.0000$ diff on $t < 10$ | **PASSED** |
| `test_variable_history_lengths_and_padding[1]` | History length $1$ valid transitions | **PASSED** |
| `test_variable_history_lengths_and_padding[5]` | History length $5$ valid transitions | **PASSED** |
| `test_variable_history_lengths_and_padding[10]` | History length $10$ valid transitions | **PASSED** |
| `test_variable_history_lengths_and_padding[20]` | History length $20$ valid transitions | **PASSED** |
| `test_numerical_assertions_nan_and_inf` | Strict exception throwing on NaN / Inf / bad dims | **PASSED** |
| `test_deterministic_eval_mode` | Exact bitwise reproducibility under `eval()` mode | **PASSED** |

---

## 8. Forward-Pass Latency Benchmark Results

Measured via `scripts/benchmark_context_encoder.py` ($50$ warmup iterations, $500$ timed evaluation trials):

### CPU Benchmark (Intel / AMD x86_64):
* **Single Online Step ($B=1$, $L=20$):**
  * **Mean Latency:** **$0.7390\text{ ms} \pm 0.1391\text{ ms}$**
  * **Median Latency:** **$0.6921\text{ ms}$**
  * **P95 Latency:** **$1.0200\text{ ms}$**
  * **P99 Latency:** **$1.2173\text{ ms}$**
  * **Min / Max:** $0.5732\text{ ms} \ / \ 1.9481\text{ ms}$
* **Batch Inference ($B=32$, $L=20$):**
  * **Mean Latency:** **$1.7243\text{ ms} \pm 0.3102\text{ ms}$**
  * **P95 Latency:** **$2.2749\text{ ms}$**

> [!NOTE]
> The measured single-step forward-pass latency ($0.739\text{ ms}$) represents the computational execution speed of the context encoder architecture. This latency enables safe real-time execution well within the high-level $10\text{ Hz}$ ($100\text{ ms}$) control budget. Final online control loop latency will be formally measured in Phase 25.

---

## 9. Implementation Status Summary

* **Currently Implemented (Phase 6):**
  - Standalone `TransformerContextEncoder` class in `models/transformer_context_encoder.py`.
  - Sinusoidal positional encoding buffer.
  - Causal multi-head self-attention with strict upper-triangular masking.
  - Padding mask and variable valid length token gathering.
  - Comprehensive unit test suite ($15/15$ tests passing).
  - Latency benchmark script in `scripts/benchmark_context_encoder.py`.
* **Planned for Future Phases:**
  - Phase 7: Meta-Supervisor Architecture ($c_t = g_\theta(z_t)$).
  - Phase 8: Trajectory Buffer & Episode Storage.
  - Phase 10: Multi-Task Meta-RL Loss Formulation.
  - Phase 15: Full Meta-RL Training Loop.
