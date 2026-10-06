# MCR-UAV: Meta-Supervisor Architecture

**Document Version:** 1.0.0  
**Project:** Meta-Contextual Reconfiguration of Hierarchical UAV Control (MCR-UAV)  
**Date:** 2026-08-12  
**Implementation Source:** `models/meta_supervisor.py`  
**Unit Tests:** `tests/test_meta_supervisor.py`  

---

## 1. Motivation

In the Meta-Contextual Reconfiguration (MCR-UAV) framework, autonomous multi-rotor flight under non-stationary dynamics and severe disturbances requires dynamic tuning of the multi-rate hierarchical control pipeline. Rather than outputting raw motor RPMs through an unconstrained black-box policy, MCR-UAV delegates low-level and mid-level stabilization to classical, verifiable controllers (Quadratic MPC and DSL PID).

The **Meta-Supervisor** ($g_\theta$) acts as an adaptive orchestrator. It receives the latent context vector $z_t \in \mathbb{R}^{16}$ (inferred by the temporal Transformer Context Encoder $f_\phi$) and outputs a strictly bounded **Multi-Tier Reconfiguration Vector** $c_t$:

$$c_t = \begin{bmatrix} \lambda_{\text{RL}} \\ \alpha_Q \\ \alpha_R \\ \alpha_P \\ \alpha_I \\ \alpha_D \\ H \end{bmatrix}$$

---

## 2. Input Latent Context Representation

The Meta-Supervisor receives:

$$z_t \in \mathbb{R}^{B \times 16}$$

where $z_t$ is the compressed temporal latent state produced by the Transformer Context Encoder from the rolling history $H_t \in \mathbb{R}^{20 \times 52}$. Input tensors must be finite float32 values; NaN and Inf inputs are strictly rejected with explicit exceptions.

---

## 3. Architecture Specification

```
                         Latent Context Vector
                         z_t ∈ R^(B × 16) or [16]
                                   │
                                   ▼
                         Shared Feature Trunk
                   Linear(16, 64) ──► LayerNorm(64) ──► GELU
                                   │
                                   ▼
                         Linear(64, 64) ──► GELU
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
     6 Continuous Parameter Heads          1 Categorical Horizon Head
          Linear(64, 1) × 6                       Linear(64, 3)
                 │                                   │
                 ▼                                   ▼
       Sigmoid Bounding Layer                     Softmax
 y_i = min + σ(x) * (max - min)             H_logits ∈ R^(B × 3)
                 │                                   │
                 ▼                                   ▼
  [λ_RL, α_Q, α_R, α_P, α_I, α_D]             Argmax / Soft-Sum
         Shape: [B, 6]                        H ∈ {10, 20, 30}
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   ▼
                       ReconfigurationVector
```

---

## 4. Multi-Tier Parameter Heads and Mathematical Mapping

### 4.1. High-Level RL Blending Authority Head ($\lambda_{\text{RL}}$)
Controls the authority blending between the high-level PPO policy action and the nominal reference path:
$$v_t^{\text{cmd}} = \lambda_{\text{RL}} v_t^{\text{RL}} + (1 - \lambda_{\text{RL}}) v_t^{\text{nominal}}$$
* **Bounding Function:** $\lambda_{\text{RL}} = 0.0 + \sigma(x_\lambda) \cdot (1.0 - 0.0) \in [0.0, 1.0]$.
* **Physical Meaning:** $\lambda_{\text{RL}} \to 1.0$ grants maximum RL authority; $\lambda_{\text{RL}} \to 0.0$ delegates full tracking to nominal baseline.

### 4.2. Mid-Level MPC Optimization Weight Scaling Heads ($\alpha_Q, \alpha_R$)
Dynamically rescales the tracking state penalty matrix $Q$ and control effort penalty matrix $R$ inside the Quadratic MPC cost function:
$$J_{\text{MPC}} = \sum_{k=0}^{H-1} \left( \|x_{t+k} - x_{t+k}^{\text{ref}}\|_{Q_t}^2 + \|u_{t+k}\|_{R_t}^2 \right)$$
$$Q_t = \alpha_Q Q_0, \quad R_t = \alpha_R R_0$$
* **$\alpha_Q$ Bounding:** $\alpha_Q = 0.2 + \sigma(x_Q) \cdot (5.0 - 0.2) \in [0.2, 5.0]$. Stiffens position tracking under strong winds.
* **$\alpha_R$ Bounding:** $\alpha_R = 0.2 + \sigma(x_R) \cdot (5.0 - 0.2) \in [0.2, 5.0]$. Penalizes violent control actions under actuator degradation.

### 4.3. Low-Level PID Tracking Gain Scaling Heads ($\alpha_P, \alpha_I, \alpha_D$)
Modulates the baseline DSL PID gains ($K_{P,0}, K_{I,0}, K_{D,0}$) at $1200\text{ Hz}$ through low-pass latched gain filters:
$$K_{P,t} = \alpha_P K_{P,0}, \quad K_{I,t} = \alpha_I K_{I,0}, \quad K_{D,t} = \alpha_D K_{D,0}$$
* **$\alpha_P$ Bounding:** $\alpha_P = 0.5 + \sigma(x_P) \cdot (2.0 - 0.5) \in [0.5, 2.0]$.
* **$\alpha_I$ Bounding:** $\alpha_I = 0.2 + \sigma(x_I) \cdot (2.5 - 0.2) \in [0.2, 2.5]$.
* **$\alpha_D$ Bounding:** $\alpha_D = 0.5 + \sigma(x_D) \cdot (2.0 - 0.5) \in [0.5, 2.0]$.

### 4.4. Discrete MPC Prediction Horizon Head ($H$)
Maps the shared representation $h_2$ to categorical logits for the precomputed MPC horizon dictionary:
$$\text{logits}_H = W_H h_2 + b_H \in \mathbb{R}^{B \times 3}$$
$$\pi_H = \text{softmax}(\text{logits}_H) \in \Delta^2$$
$$H_t = \begin{cases} 10, & \text{if } \operatorname{argmax}(\text{logits}_H) = 0 \\ 20, & \text{if } \operatorname{argmax}(\text{logits}_H) = 1 \\ 30, & \text{if } \operatorname{argmax}(\text{logits}_H) = 2 \end{cases}$$
* **Discrete Guarantee:** In hard deployment and inference mode (`return_hard=True`), $H_t$ is strictly restricted to the discrete dictionary $\{10, 20, 30\}$. Fractional or unconstrained horizons (e.g. $17, 24.6$) are mathematically prohibited.
* **Differentiable Training Proxy:** During Meta-RL policy training (`return_hard=False`), the supervisor computes the expected horizon $\mathbb{E}[H] = \sum_{i=1}^3 \pi_i H_i$ as a differentiable gradient proxy. This soft expectation is exclusively used for backpropagation and is never passed to the physical MPC solver during flight.

> [!IMPORTANT]
> The configured parameter bounds ($\lambda_{\text{RL}} \in [0.0, 1.0]$, $\alpha_Q, \alpha_R \in [0.2, 5.0]$, $\alpha_P, \alpha_D \in [0.5, 2.0]$, $\alpha_I \in [0.2, 2.5]$, $H \in \{10, 20, 30\}$) represent **empirically motivated design constraints and bounding safety mechanisms**, rather than formal analytical closed-loop Lyapunov stability guarantees. Closed-loop stability, gain transition smoothness, and dynamic tracking stability will be systematically evaluated through hardware-in-the-loop and simulated benchmark sweeps in subsequent integration phases.

---

## 5. Authoritative Parameter Bounds Summary

| Parameter | Notation | Target Range | Nominal Value | Safety & Stability Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **RL Blending Authority** | $\lambda_{\text{RL}}$ | $[0.0, 1.0]$ | $1.0$ | Sigmoid scaling; smooth interpolation between RL and nominal path. |
| **MPC State Weight Scale** | $\alpha_Q$ | $[0.2, 5.0]$ | $1.0$ | Sigmoid scaling; prevents ill-conditioned QP Hessian matrices. |
| **MPC Control Weight Scale** | $\alpha_R$ | $[0.2, 5.0]$ | $1.0$ | Sigmoid scaling; prevents control input saturation under gusts. |
| **PID Proportional Gain Scale** | $\alpha_P$ | $[0.5, 2.0]$ | $1.0$ | Sigmoid scaling; prevents loop-gain instability while allowing stiffening. |
| **PID Integral Gain Scale** | $\alpha_I$ | $[0.2, 2.5]$ | $1.0$ | Sigmoid scaling; prevents limit-cycle integral windup during tracking offsets. |
| **PID Derivative Gain Scale** | $\alpha_D$ | $[0.5, 2.0]$ | $1.0$ | Sigmoid scaling; prevents derivative noise amplification. |
| **MPC Prediction Horizon** | $H$ | $\{10, 20, 30\}$ | $20$ | Categorical argmax; guarantees compatibility with precomputed system matrices. |

---

## 6. Trainable Parameter Count Breakdown

| Component | Layer / Head | Trainable Parameters |
| :--- | :--- | :---: |
| **Trunk Linear 1** | `Linear(16, 64)` | $16 \times 64 + 64 = 1,088$ |
| **Trunk LayerNorm** | `LayerNorm(64)` | $64 \times 2 = 128$ |
| **Trunk Linear 2** | `Linear(64, 64)` | $64 \times 64 + 64 = 4,160$ |
| **$\lambda_{\text{RL}}$ Head** | `Linear(64, 1)` | $64 \times 1 + 1 = 65$ |
| **$\alpha_Q$ Head** | `Linear(64, 1)` | $64 \times 1 + 1 = 65$ |
| **$\alpha_R$ Head** | `Linear(64, 1)` | $64 \times 1 + 1 = 65$ |
| **$\alpha_P$ Head** | `Linear(64, 1)` | $64 \times 1 + 1 = 65$ |
| **$\alpha_I$ Head** | `Linear(64, 1)` | $64 \times 1 + 1 = 65$ |
| **$\alpha_D$ Head** | `Linear(64, 1)` | $64 \times 1 + 1 = 65$ |
| **Horizon $H$ Head** | `Linear(64, 3)` | $64 \times 3 + 3 = 195$ |
| **Total Trainable Parameters** | **Entire Meta-Supervisor** | **$5,961$** |

---

## 7. Unit Test Results (`tests/test_meta_supervisor.py`)

All $18$ unit test suites passed with $100\%$ success rate:

| Test Name | Verified Condition | Status |
| :--- | :--- | :---: |
| `test_architecture_configuration` | Correct latent dim ($16$), hidden dim ($64$), $3$ horizon classes | **PASSED** |
| `test_parameter_count_and_breakdown` | Exact total ($5,961$) and layer-by-layer breakdown | **PASSED** |
| `test_batch_sizes_and_output_shapes[1]` | Batch size $B=1 \to$ All tensors shape $[1, \cdot]$ | **PASSED** |
| `test_batch_sizes_and_output_shapes[4]` | Batch size $B=4 \to$ All tensors shape $[4, \cdot]$ | **PASSED** |
| `test_batch_sizes_and_output_shapes[8]` | Batch size $B=8 \to$ All tensors shape $[8, \cdot]$ | **PASSED** |
| `test_batch_sizes_and_output_shapes[16]` | Batch size $B=16 \to$ All tensors shape $[16, \cdot]$ | **PASSED** |
| `test_batch_sizes_and_output_shapes[32]` | Batch size $B=32 \to$ All tensors shape $[32, \cdot]$ | **PASSED** |
| `test_single_sample_unbatched_1d_input` | Unbatched input $[16] \to$ Output shape $[1, \cdot]$ | **PASSED** |
| `test_deterministic_inference` | Repeated forward calls produce exact bitwise identity | **PASSED** |
| `test_gradient_flow_continuous_and_discrete` | Non-zero, finite gradients on all continuous heads and horizon head | **PASSED** |
| `test_nan_input_rejection` | ValueError raised on NaN input tensor | **PASSED** |
| `test_inf_input_rejection` | ValueError raised on Inf input tensor | **PASSED** |
| `test_feature_dim_mismatch_rejection` | ValueError raised on dimension mismatch ($z \ne 16$) | **PASSED** |
| `test_continuous_parameter_bound_compliance` | $100\%$ compliance across $100$ random latent vectors | **PASSED** |
| `test_extreme_latent_boundary_robustness` | Finite bounded outputs on $z = \mathbf{0}, z = +1000, z = -1000$ | **PASSED** |
| `test_horizon_categorical_validity` | $H \in \{10, 20, 30\}$ strictly across $500$ random trials | **PASSED** |
| `test_serialization_and_state_dict_consistency` | Save/load state_dict produces exact bitwise identity | **PASSED** |
| `test_to_numpy_export_helper` | Clean export to NumPy dictionaries for controller integration | **PASSED** |

---

## 8. Implementation Status Summary

* **Currently Implemented (Phase 7):**
  - Standalone `MetaSupervisor` class in `models/meta_supervisor.py`.
  - Configurable `ReconfigurationBounds` and structured `ReconfigurationVector` dataclasses.
  - Six bounded continuous parameter heads and one categorical horizon head.
  - Comprehensive unit test suite ($18/18$ passing unit tests).
  - Total full-workspace unit tests: $51/51$ passing.
* **Planned for Future Phases:**
  - Phase 8: Trajectory Buffer & Episode Storage.
  - Phase 9: Hierarchical Environment Reconfiguration Integration ($c_t \to \text{RL/MPC/PID}$).
  - Phase 10: Multi-Task Meta-RL Loss Formulation.
  - Phase 15: Meta-RL Training Loop.
