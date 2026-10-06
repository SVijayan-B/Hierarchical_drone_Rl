# Hierarchical Environment Reconfiguration Integration

This document outlines the architecture and integration of the Phase-7 **Meta-Supervisor** output into the live hierarchical UAV control pipeline (consisting of high-level PPO waypoint guidance, mid-level Model Predictive Control (MPC) trajectory generation, and low-level Proportional-Integral-Derivative (PID) stabilization).

---

## 1. Reconfiguration Vector Definition

The Meta-Supervisor produces a 7-dimensional command vector $c_t$ at a frequency of 10 Hz:

$$c_t = [\lambda_{\text{RL}}, \alpha_Q, \alpha_R, \alpha_P, \alpha_I, \alpha_D, H]^T$$

Where:
*   $\lambda_{\text{RL}} \in [0.0, 1.0]$: Blending weight for high-level guidance.
*   $\alpha_Q \in [0.2, 5.0]$: Scaling factor for MPC state cost matrix $Q$.
*   $\alpha_R \in [0.2, 5.0]$: Scaling factor for MPC control effort cost matrix $R$.
*   $\alpha_P \in [0.5, 2.0]$: Tuning multiplier for the PID proportional gain $K_p$.
*   $\alpha_I \in [0.2, 2.5]$: Tuning multiplier for the PID integral gain $K_i$.
*   $\alpha_D \in [0.5, 2.0]$: Tuning multiplier for the PID derivative gain $K_d$.
*   $H \in \{10, 20, 30\}$: Hard discrete horizon length for MPC.

---

## 2. Multi-Rate Reconfiguration Pipeline

The MCR-UAV controller operates as a multi-rate hierarchical pipeline:
1.  **PPO Waypoint Planner & Meta-Supervisor (10 Hz):** Evaluates environment context and publishes high-level commands and reconfiguration parameters.
2.  **MPC Controller (10 Hz):** Dynamically scales cost matrices and updates prediction horizons discretely.
3.  **Low-Level PID Controller (1200 Hz):** Runs in PyBullet's inner step loop. It interpolates target PID gains exponentially without neural network inference.

### 2.1 Multi-Rate Latching Mechanism
To preserve execution frequency separation and avoid neural network inference overhead in the high-frequency inner loops:
*   The supervisor is only called at 10 Hz.
*   Reconfiguration vectors $c_t$ are **latched** at 10 Hz.
*   The low-level inner loop uses the latched multipliers to adjust PID gains continuously at 1200 Hz.

### 2.2 Low-Level Gain Smoothing (1200 Hz)
To prevent step discontinuities in actuator outputs and avoid mechanical vibration or unstable control loops, PID gains are smoothed exponentially at 1200 Hz using $\beta_{\text{gain}} = 0.05$:

$$K_{\text{new}} = (1 - \beta_{\text{gain}}) K_{\text{old}} + \beta_{\text{gain}} K_{\text{target}}$$

$$\alpha_{\text{smoothed}}^{(k)} = 0.95 \alpha_{\text{smoothed}}^{(k-1)} + 0.05 \alpha_{\text{target}}$$

This smoothing applies independently to $\alpha_P$, $\alpha_I$, and $\alpha_D$ multipliers.

---

## 3. Controller Modifications

### 3.1 High-Level Command Blending
High-level PPO waypoint commands and global A* guidance coordinates are blended continuously using the reconfigured blending weight:

$$\text{target\_wp} = \lambda_{\text{RL}} \cdot \text{rl\_wp} + (1.0 - \lambda_{\text{RL}}) \cdot \text{astar\_wp}$$

If reconfiguration is disabled or invalid, it falls back to the baseline blending:

$$\lambda_{\text{RL}} = 1.0 - \text{guidance\_blend}$$

### 3.2 MPC Cost Scaling & Horizon Switching
The Model Predictive Controller modifies cost matrices and horizons dynamically:

$$Q_d = \alpha_Q \cdot Q_0$$
$$R_d = \alpha_R \cdot R_0$$

Where $Q_0$ and $R_0$ represent the base nominal cost matrices.

The prediction horizon switches discretely among pre-allocated matrix stacks:

$$H \in \{10, 20, 30\}$$

Discrete horizon switching uses precomputed prediction matrices $M$ and $C$ to ensure zero-overhead runtime reconfiguration.

### 3.3 Low-Level PID Gains & Anti-Windup
PID coefficients are updated continuously at 1200 Hz:

$$P_{\text{coeff}} = P_{\text{coeff, base}} \cdot \alpha_P$$
$$I_{\text{coeff}} = I_{\text{coeff, base}} \cdot \alpha_I$$
$$D_{\text{coeff}} = D_{\text{coeff, base}} \cdot \alpha_D$$

Low-level actuator limits and integrator anti-windup bounds are preserved exactly to maintain physical safety.

---

## 4. Fallback nominal configuration
If any parameter in the reconfiguration vector $c_t$ is invalid (out-of-bounds, `NaN`, or `inf`), the system falls back to the nominal configuration:

$$\lambda_{\text{RL}} = 1.0, \quad \alpha_Q = 1.0, \quad \alpha_R = 1.0, \quad \alpha_P = 1.0, \quad \alpha_I = 1.0, \quad \alpha_D = 1.0, \quad H = 20$$

---

## 5. Verification results
The integration has been fully verified using a suite of 26 unit and regression tests in [test_reconfiguration_controller.py](file:///d:/IT%20SECTOR/PROJECTS/Drone/Drone3/RL-Drone/tests/test_reconfiguration_controller.py):
*   **Nominal Resets & Regressions:** Verified that resetting the environment clears all active configurations and restores exact nominal behaviors.
*   **Validation & Rejection:** NaN, Inf, and out-of-bounds inputs are correctly rejected, triggering safe fallback to nominal configurations.
*   **Dynamic Scaling:** Q/R matrices are verified to scale proportionally, and the discrete horizon switches dynamically without compilation or runtime overhead.
*   **Latching & Smoothing:** Confirmed that supervisor parameters are latched at 10 Hz, and PID gains converge exponentially at 1200 Hz.
*   **Anti-windup & Multirate Latching:** Integrated PID controller continues to clip accumulated errors safely, and no supervisor updates occur in the inner loop.
