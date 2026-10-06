# MCR-UAV: Formal Research Formulation and Experimental Protocol

**Document Version:** 1.0.0  
**Project:** Meta-Contextual Reconfiguration of Hierarchical UAV Control for Rapid Adaptation to Unseen Disturbances (MCR-UAV)  
**Date:** 2026-08-12  
**Target Venue:** IEEE Transactions on Robotics (T-RO) / IEEE Robotics and Automation Letters (RA-L)  
**Status:** Formal Research Problem, Hypotheses, Mathematical Model, and Experimental Protocol  

---

## 1. Research Problem

Autonomous quadrotor unmanned aerial vehicles (UAVs) operating in complex, dynamic, and unstructured environments are subject to severe unmodeled physical variations. These variations include non-stationary turbulent wind gusts, significant sensor noise attacks, ground effects, mass/inertia shifts, and severe asymmetric actuator loss-of-effectiveness (LoE) caused by rotor damage or battery voltage drops.

Existing reinforcement learning (RL) and control approaches suffer from a fundamental trade-off:
1. **Monolithic End-to-End Deep RL Policies** (e.g., Kaufmann et al., Nature 2023; Bauersfeld et al., IEEE ICRA 2024; Zhang et al., arXiv 2026 [MAVEN]) map sensor observations directly to motor commands or body rates. While agile under nominal conditions, they lack deterministic constraint satisfaction, provide no transparent autopilot fallbacks, and fail catastrophically when encountering structural faults or dynamics far outside their training distribution.
2. **Classical and Learning-Based Single-Layer Adaptive Control** (e.g., O'Connell et al., Science Robotics 2022 [Neural-Fly]; Salzmann et al., Science Robotics 2023; Chee et al., IEEE CDC 2022) adapts only an isolated single tier (e.g., an additive feedforward aerodynamic wrench or isolated PID gains). They cannot simultaneously adjust mid-level predictive planning horizons, state/input cost weights, and low-level stabilization damping.
3. **Online Gradient-Based Meta-RL** (e.g., Bellegarda & Nguyen, IEEE RA-L 2022; Nagabandi et al., ICLR 2019) requires iterative backpropagation during flight, which introduces unacceptable computational latency ($>50\text{ ms}$) and risk of divergence on micro-aerial vehicle avionics.

**The Exact Scientific Problem Addressed:**  
*How can an autonomous UAV infer its latent physical operating conditions from a short sequence of onboard transitions in a single forward pass, and dynamically reconfigure the parameters, authority, and optimization horizon of an explicit hierarchical $\text{RL} \to \text{MPC} \to \text{PID}$ control stack to achieve rapid adaptation to compound, out-of-distribution disturbances and actuator degradations without online gradient descent, policy retraining, or compromising nominal flight stability?*

### Distinction from the Existing Honors Baseline
* **Existing Honors Baseline:** A fixed hierarchical architecture where a temporal Transformer PPO policy generates 3D velocity commands, a Quadratic MPC tracks waypoints with static cost weights and fixed/heuristic horizons, and low-level DSL PID uses static gains with heuristic scaling. As demonstrated in Phase 0 smoke tests, this baseline achieves $0.0\%$ success under asymmetric motor degradation ($30\%$ LoE on rotors 1 and 2).
* **Proposed MCR-UAV Framework:** Introduces a dedicated **Transformer Context Encoder** $f_\phi$ inferring latent condition $z_t$, coupled to a **Meta-Reconfiguration Supervisor** $g_\theta$ that outputs a dynamic multi-tier reconfiguration vector $c_t = [\lambda_{\text{RL}}, \alpha_Q, \alpha_R, \alpha_P, \alpha_I, \alpha_D, H_{\text{MPC}}]$ to adapt high-level authority, mid-level MPC optimization, and low-level PID damping in real time.

---

## 2. Research Objectives

### Primary Objective
To design, implement, and empirically validate a meta-contextual controller-reconfiguration architecture (**MCR-UAV**) that enables an autonomous quadrotor to rapidly infer unobserved operating conditions from short interaction sequences and dynamically adapt an explicit hierarchical control pipeline ($\text{RL} \to \text{MPC} \to \text{PID}$) to unseen aerodynamic disturbances, sensor corruption, and severe actuator degradation in simulation without online backpropagation.

### Secondary Objectives
1. **Context Representation Objective:** Develop a temporal self-attention Transformer Context Encoder $f_\phi(H_t)$ that extracts an informative, low-dimensional latent embedding $z_t \in \mathbb{R}^{d_z}$ from short transition histories $H_t = [s_{t-N:t}, a_{t-N:t-1}, r_{t-N:t-1}, \Delta s_{t-N:t}]$ without future information leakage.
2. **Multi-Tier Reconfiguration Objective:** Construct a Meta-RL Reconfiguration Policy $g_\theta(z_t)$ that maps latent context $z_t$ into bounded, provably safe adjustments for high-level RL blending authority ($\lambda_{\text{RL}}$), mid-level discrete MPC horizon ($H \in \{10, 20, 30\}$) and cost weights ($Q, R$), and low-level PID gains ($K_P, K_I, K_D$).
3. **Actuator Degradation Objective:** Investigate whether multi-tier reconfiguration improves trajectory tracking and survival rates under progressive asymmetric rotor loss-of-effectiveness ($0\%$ to $70\%$ LoE) compared to conventional domain-randomized baselines.
4. **Generalization & Adaptation Objective:** Characterize zero-shot and few-shot adaptation performance across out-of-distribution (OOD) wind profiles, sudden physical impulses, and multi-modal compound disturbances.
5. **Avionics & Real-Time Feasibility Objective:** Formulate and benchmark the multi-rate timing architecture ($10\text{ Hz}$ meta-reconfiguration / MPC with $1200\text{ Hz}$ inner-loop PID) to ensure forward-pass inference latency remains strictly within embedded control execution budgets.

---

## 3. Formal Research Questions

* **RQ1 (Robustness and Generalization to Unseen/OOD Disturbances):**  
  *Does forward-pass meta-contextual reconfiguration improve trajectory tracking accuracy, disturbance recovery time, and episode success rates under unseen aerodynamic gusts, sensor corruption, and compound out-of-distribution physical conditions compared to static hierarchical baselines and conventional domain-randomized Transformer policies?*

* **RQ2 (Benefit of Multi-Tier Simultaneous Reconfiguration):**  
  *Does simultaneous, coordinated reconfiguration of all three hierarchical tiers (high-level authority $\lambda_{\text{RL}}$, mid-level MPC parameters $\{H, Q, R\}$, and low-level PID gains $\{K_P, K_I, K_D\}$) yield statistically significant performance and stability gains over single-tier adaptation baselines (RL-only, MPC-only, or PID-only adaptation)?*

* **RQ3 (Rapid Adaptation Dynamics without Online Optimization):**  
  *Can forward-pass latent context inference ($z_t = f_\phi(H_t)$) achieve rapid parameter convergence and recovery within an interaction budget of $\le 10$ control steps ($1.0\text{ s}$) following an abrupt disturbance onset, without requiring iterative online gradient descent or policy weight updates?*

---

## 4. Testable Hypotheses

* **H1 (Out-of-Distribution Disturbance Robustness):**  
  *Under held-out and out-of-distribution (OOD) wind, gust, and sensor noise conditions, MCR-UAV will achieve higher trajectory success rates and lower root-mean-square tracking error (RMSE) than non-adaptive baselines (B1–B5) and the domain-randomized Transformer baseline (B6).*

* **H2 (Actuator Degradation Tolerance):**  
  *Under asymmetric rotor thrust loss ($\ge 30\%$ LoE on rotors 1 and 2 where static baselines B1–B6 achieve $0.0\%$ success), MCR-UAV will achieve non-zero success and bounded tracking error by dynamically reducing RL authority and adjusting PID/MPC damping.*

* **H3 (Multi-Tier vs. Single-Tier Synergy):**  
  *Simultaneous multi-tier reconfiguration (A4: Full MCR-UAV) will achieve faster settling times, lower tracking RMSE, and lower peak jerk than isolated single-tier adaptation ablations (A1: RL-only, A2: MPC-only, A3: PID-only).*

* **H4 (Rapid Forward-Pass Adaptation Speed):**  
  *Following an abrupt physical disturbance onset (impulse or sudden degradation), MCR-UAV will adapt its control parameters and initiate trajectory recovery within $N_{\text{adapt}} \le 10$ RL steps ($1.0\text{ s}$) purely via feedforward context inference.*

* **H5 (Nominal Performance Preservation):**  
  *Under nominal conditions (Scenario A: No Wind / No Faults), MCR-UAV will not exhibit statistically significant degradation in tracking RMSE ($p > 0.05$), total energy proxy, or trajectory smoothness compared to the nominal baseline B6.*

> [!NOTE]  
> None of the above hypotheses are assumed to be true *a priori*. Each hypothesis will be subjected to rigorous empirical evaluation across $\ge 5$ random seeds with matched test splits.

---

## 5. Formal Mathematical Model

```
                    ┌────────────────────────────────────────────────────────┐
                    │  Onboard Transition Sequence:                          │
                    │  H_t = [s_{t-N:t}, a_{t-N:t-1}, r_{t-N:t-1}, Δs_{t-N:t}]│
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │      Transformer Context Encoder: z_t = f_φ(H_t)       │
                    │    - Latent Condition Vector: z_t ∈ ℝ^{d_z}            │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │       Meta-Supervisor Policy: c_t = g_θ(z_t)           │
                    │   c_t = [λ_RL, α_Q, α_R, α_P, α_I, α_D, H_MPC]        │
                    └───────┬───────────────────┼────────────────────┬───────┘
                            │                   │                    │
                            ▼                   ▼                    ▼
                    ┌───────────────┐   ┌───────────────┐    ┌───────────────┐
                    │ High-Level    │   │ Mid-Level     │    │ Low-Level     │
                    │ RL Authority: │   │ Adaptive MPC: │    │ Adaptive PID: │
                    │ v_cmd =       │   │ H = H_MPC     │    │ K_P = α_P K_P0│
                    │ λ_RL v_RL +   │   │ Q = α_Q Q_0   │    │ K_I = α_I K_I0│
                    │ (1-λ_RL)v_MPC │   │ R = α_R R_0   │    │ K_D = α_D K_D0│
                    └───────────────┘   └───────────────┘    └───────────────┘
```

### 5.1 State, Action, and Transition Definitions
Let $t \in \mathbb{N}$ denote the discrete decision step at the high/mid-level decision frequency ($f_{\text{RL}} = 10\text{ Hz}$, $\Delta t_{\text{RL}} = 0.1\text{ s}$).

* **State Observation ($s_t \in \mathbb{R}^{d_s}$):**
  $$s_t = \big[\mathbf{p}_t, \mathbf{v}_t, \mathbf{q}_t, \boldsymbol{\omega}_t, (\mathbf{p}_t^{\text{target}} - \mathbf{p}_t), \|\mathbf{p}_t^{\text{target}} - \mathbf{p}_t\|_2, \mathbf{d}_t^{\text{ultra}}, \mathbf{a}_{t-1}\big]$$
  where $\mathbf{p}_t \in \mathbb{R}^3$ is position, $\mathbf{v}_t \in \mathbb{R}^3$ is linear velocity, $\mathbf{q}_t \in \mathbb{R}^3$ is Euler attitude (roll, pitch, yaw), $\boldsymbol{\omega}_t \in \mathbb{R}^3$ is angular rate, $(\mathbf{p}_t^{\text{target}} - \mathbf{p}_t) \in \mathbb{R}^3$ is target relative vector, $\|\mathbf{p}_t^{\text{target}} - \mathbf{p}_t\|_2 \in \mathbb{R}^1$ is target distance, $\mathbf{d}_t^{\text{ultra}} \in \mathbb{R}^5$ is the 5-beam ultrasonic distance measurement array, and $\mathbf{a}_{t-1} \in \mathbb{R}^3$ is the previous velocity command ($d_s = 3 + 3 + 3 + 3 + 3 + 1 + 5 + 3 = 24$).

* **State Increment ($\Delta s_t \in \mathbb{R}^{d_s}$):**
  $$\Delta s_t = s_t - s_{t-1} \in \mathbb{R}^{24}$$

* **Action ($\mathbf{a}_t \in \mathbb{R}^3$):** High-level desired 3D velocity setpoint $\mathbf{v}_t^{\text{des}} \in [-v_{\max}, v_{\max}]^3$ ($d_a = 3$).

* **Reward ($r_t \in \mathbb{R}$):**
  $$r_t = - w_p \|\mathbf{p}_t - \mathbf{p}_t^{\text{ref}}\|_2 - w_v \|\mathbf{v}_t - \mathbf{v}_t^{\text{des}}\|_2 - w_a \|\mathbf{a}_t - \mathbf{a}_{t-1}\|_2^2 - w_\omega \|\boldsymbol{\omega}_t\|_2^2 + R_{\text{target}} \cdot \mathbb{I}_{\text{reach}}$$

### 5.2 Temporal History and Context Inference
* **Interaction History ($H_t \in \mathbb{R}^{N \times d_h}$):** For a sequence history length $N=20$:
  $$H_t = \Big\{ \big(s_{t-k}, \mathbf{a}_{t-k-1}, r_{t-k-1}, \Delta s_{t-k}\big) \Big\}_{k=0}^{N-1}$$
  where $d_h = d_s + d_a + 1 + d_s = 24 + 3 + 1 + 24 = 52$.

* **Transformer Context Encoder ($f_\phi$):**
  $$\mathbf{z}_t = f_\phi(H_t) = \text{LayerNorm}\Big(\text{MultiHeadAttention}\big(\text{Embed}(H_t)\big)\Big)_{[-1, :]} \in \mathbb{R}^{d_z}$$
  where $d_z = 32$ is the latent context embedding dimension.

* **Meta-Supervisor Policy ($g_\theta$):**
  $$\mathbf{c}_t = g_\theta(\mathbf{z}_t) = \sigma_{\text{bounded}}\big(\text{MLP}_\theta(\mathbf{z}_t)\big) \in \mathbb{R}^7$$

### 5.3 Controller Reconfiguration Equations
The reconfiguration vector is defined as:
$$\mathbf{c}_t = \big[\lambda_{\text{RL}, t}, \, \alpha_{Q, t}, \, \alpha_{R, t}, \, \alpha_{P, t}, \, \alpha_{I, t}, \, \alpha_{D, t}, \, H_{\text{MPC}, t}\big]^T$$

1. **High-Level RL Authority Blending:**
   $$\mathbf{v}_t^{\text{cmd}} = \lambda_{\text{RL}, t} \mathbf{v}_t^{\text{RL}} + (1 - \lambda_{\text{RL}, t}) \mathbf{v}_t^{\text{MPC}}$$
   where $\lambda_{\text{RL}, t} \in [\lambda_{\min}, \lambda_{\max}] \subseteq [0.0, 1.0]$.

2. **Mid-Level MPC Optimization Reconfiguration:**
   The Quadratic MPC minimizes:
   $$J(\mathbf{U}) = \sum_{k=1}^{H_t} \Big[ (\mathbf{x}_{t+k} - \mathbf{x}_{t+k}^{\text{ref}})^T Q_t (\mathbf{x}_{t+k} - \mathbf{x}_{t+k}^{\text{ref}}) + \mathbf{u}_{t+k-1}^T R_t \mathbf{u}_{t+k-1} \Big]$$
   subject to $\mathbf{x}_{k+1} = A \mathbf{x}_k + B \mathbf{u}_k$, $\|\mathbf{v}\| \le v_{\max}$, $\|\mathbf{u}\| \le a_{\max}$.
   * Horizon Reconfiguration: $H_t = \arg\min_{h \in \{10, 20, 30\}} |h - H_{\text{MPC}, t}|$.
   * Cost Matrices: $Q_t = \alpha_{Q, t} Q_0$, $R_t = \alpha_{R, t} R_0$.

3. **Low-Level Cascaded PID Gain Reconfiguration:**
   $$\mathbf{K}_{P, t} = \alpha_{P, t} \mathbf{K}_{P, 0}, \quad \mathbf{K}_{I, t} = \alpha_{I, t} \mathbf{K}_{I, 0}, \quad \mathbf{K}_{D, t} = \alpha_{D, t} \mathbf{K}_{D, 0}$$
   where $\alpha_{P, t}, \alpha_{I, t}, \alpha_{D, t} \in [\alpha_{\min}, \alpha_{\max}] \subset \mathbb{R}^+$.

> [!IMPORTANT]  
> **Status Distinction:** The equations for $s_t$, Quadratic MPC ($A, B$), and baseline PID are *already implemented* in the frozen baseline. The equations for $H_t$, $f_\phi$, $g_\theta$, $\lambda_{\text{RL}, t}$, and $\alpha(z_t)$ are *design equations* to be implemented in Phases 6–11.

---

## 6. Multi-Rate Timing and Safety Architecture

```
  High-Level Layer (10 Hz, Δt = 100 ms):
  [Transition Buffer] ──► [Transformer Context f_φ] ──► [Meta-Supervisor g_θ] ──► c_t
                                                                                    │
  Mid-Level Layer (10 Hz, Δt = 100 ms):                                             │
  [PPO Policy] + [Adaptive Quadratic MPC (H_t, Q_t, R_t)] ◄────────────────────────┼── (λ_RL, H, Q, R)
                                │                                                   │
                                ▼ Desired (v_x, v_y, v_z)                           │
  Low-Level Layer (1200 Hz, Δt = 0.833 ms):                                         │
  [Zero-Order Hold / Filter] ──► [Upgraded DSL PID (K_P(t), K_I(t), K_D(t))] ◄──────┴── (α_P, α_I, α_D)
                                │
                                ▼ Motor RPMs (ω_1, ω_2, ω_3, ω_4)
                         [PyBullet Quadrotor Dynamics]
```

### Multi-Rate Timing Interaction
* **Meta-Supervisor & High/Mid-Level Control:** Executes at $f_{\text{RL}} = 10\text{ Hz}$ ($\Delta t = 100\text{ ms}$).
* **Low-Level Inner Loop Stabilization:** Executes at $f_{\text{PID}} = 1200\text{ Hz}$ ($\Delta t = 0.833\text{ ms}$).
* **Safe Asynchronous Interfacing:**
  1. *Parameter Latching (Zero-Order Hold):* Reconfiguration parameters $c_t$ are updated at the $10\text{ Hz}$ boundary and latched across the subsequent $120$ PID sub-steps.
  2. *Low-Pass Gain Transition Filter:* To prevent step discontinuities in PID control signals upon gain switching, gains follow an exponential smoothing filter:
     $$\mathbf{K}_{\text{applied}}[k] = (1 - \beta_{\text{gain}}) \mathbf{K}_{\text{applied}}[k-1] + \beta_{\text{gain}} \mathbf{K}_{t}, \quad \beta_{\text{gain}} = 0.05$$
  3. *Derivative Noise Filtering:* As established in Phase 0, Euler angle rates are filtered with $\alpha_{\text{deriv}} = 0.04$ to eliminate high-frequency chatter.
  4. *Anti-Windup Integration Freezing:* Integrator accumulation is paused when actuator saturation occurs:
     $$\dot{\mathbf{e}}_I = \mathbf{0} \quad \text{if } \|\mathbf{u}_{\text{raw}}\| > u_{\max} \text{ and } \text{sign}(\mathbf{u}_{\text{raw}}) = \text{sign}(\mathbf{e})$$

---

## 7. Parameter Bounds and Constraint Specifications

| Reconfiguration Parameter | Symbol | Allowed Range | Nominal Baseline | Physical Justification / Bounding Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **RL Blending Authority** | $\lambda_{\text{RL}}$ | $[0.0, 1.0]$ | $1.0$ (RL only) | Sigmoid activation; bounds policy authority between pure MPC ($\lambda=0$) and pure RL ($\lambda=1$). |
| **MPC Prediction Horizon** | $H_{\text{MPC}}$ | $\{10, 20, 30\}$ | $20$ | Discrete argmin mapping; restricts prediction steps to precomputed matrix dictionary $M_h, C_h$. |
| **MPC State Weight Scale** | $\alpha_Q$ | $[0.2, 5.0]$ | $1.0$ | Clamped scaling; prevents ill-conditioned Hessian in QP quadratic solver. |
| **MPC Input Weight Scale** | $\alpha_R$ | $[0.2, 5.0]$ | $1.0$ | Clamped scaling; prevents excessive acceleration commands under high disturbance. |
| **PID Proportional Gain Scale** | $\alpha_P$ | $[0.5, 2.0]$ | $1.0$ | Clamped scaling; prevents loop gain instability while allowing stiffening under wind. |
| **PID Integral Gain Scale** | $\alpha_I$ | $[0.2, 2.5]$ | $1.0$ | Clamped scaling; prevents integral windup limit cycles during large tracking offsets. |
| **PID Derivative Gain Scale** | $\alpha_D$ | $[0.5, 2.0]$ | $1.0$ | Clamped scaling; prevents derivative noise amplification. |

> [!NOTE]  
> Specific minimum and maximum bounds are **DESIGN PARAMETERS TO BE EMPIRICALLY REFINED IN PHASE 14 (Meta-Validation)**.

---

## 8. Independent Variables

The independent variables manipulated during systematic evaluation are:

1. **Controller Architecture:** (Baselines B1–B6, Ablations A0–A4, and MCR-UAV).
2. **Disturbance Modality:** (Wind drag, periodic gusts, impulse forces, sensor corruption, motor degradation, and compound multi-modal disturbances).
3. **Aerodynamic Wind Magnitude ($v_w$):** Continuous range $[0.0, 6.0]\text{ m/s}$.
4. **Gust Frequency ($f_w$):** Range $[0.5, 5.0]\text{ Hz}$.
5. **Impulse Force Magnitude ($F_{\text{imp}}$):** $[0.0, 6.0]\text{ N}$ applied for $\Delta t_{\text{imp}} = 0.1\text{ s}$ ($1$ RL step).
6. **Sensor Noise Multiplication Factor ($k_{\text{noise}}$):** $[1.0\times, 5.0\times]$ standard IMU and ultrasonic noise variance.
7. **Actuator Loss-of-Effectiveness ($\delta_{\text{motor}}$):** Severity sweep $[0\%, 10\%, 20\%, 30\%, 40\%, 50\%, 60\%, 70\%]$ on rotors 1 and 2.
8. **Random Evaluation Seeds:** Fixed set of 5 evaluation seeds: $\{11, 42, 101, 777, 2026\}$.

---

## 9. Dependent Variables (Evaluation Metrics)

| Metric | Unit | Mathematical Definition | Primary Evaluation Role |
| :--- | :---: | :--- | :--- |
| **Success Rate ($\text{SR}$)** | $\%$ | $\frac{1}{M}\sum_{m=1}^M \mathbb{I}(\text{Target Reached \& Hovered } \ge 2\text{s})$ | Primary task completion metric. |
| **Tracking RMSE** | $\text{m}$ | $\sqrt{\frac{1}{T}\sum_{t=1}^T \|\mathbf{p}_t - \mathbf{p}_t^{\text{ref}}\|_2^2}$ | Trajectory tracking accuracy. |
| **Mean Tracking Error** | $\text{m}$ | $\frac{1}{T}\sum_{t=1}^T \|\mathbf{p}_t - \mathbf{p}_t^{\text{ref}}\|_2$ | Average path deviation. |
| **Settling Time ($T_s$)** | $\text{s}$ | $\min \{ t \mid \|\mathbf{p}_\tau - \mathbf{p}^{\text{target}}\|_2 \le 0.20\text{ m}, \forall \tau \ge t \}$ | Convergence speed to target. |
| **Impulse Recovery Time ($T_{\text{rec}}$)**| $\text{s}$ | Time to return to within $0.15\text{ m}$ of reference trajectory post-impulse. | Transient disturbance rejection. |
| **Peak Jerk** | $\text{m/s}^3$ | $\max_t \|\dddot{\mathbf{p}}_t\|_2$ | Mechanical stress and chatter. |
| **Average Jerk** | $\text{m/s}^3$ | $\frac{1}{T}\sum_{t=1}^T \|\dddot{\mathbf{p}}_t\|_2$ | Flight smoothness. |
| **Total Energy Proxy** | $\text{RPM}^2$ | $\sum_{t=1}^T \sum_{i=1}^4 \omega_{i, t}^2$ | Actuator effort proxy. |
| **Energy per Meter** | $\text{RPM}^2/\text{m}$ | $\text{Total Energy} / \text{Path Length}$ | Transport efficiency. |
| **Control Effort** | $-$ | $\frac{1}{T}\sum_{t=1}^T \|\Delta \mathbf{u}_t\|_2^2$ | Actuator wear / variation. |
| **Trajectory Smoothness** | $-$ | $\int_0^T \|\ddot{\mathbf{p}}(t)\|_2^2 dt$ | Acceleration integral. |
| **Inference Latency** | $\text{ms}$ | Wall-clock execution time of forward pass ($f_\phi + g_\theta$). | Embedded computational cost. |
| **Adaptation Steps ($N_{\text{adapt}}$)**| steps | Number of RL steps until tracking error enters steady-state band post-fault. | Rapid adaptation speed. |
| **Context Variance** | $-$ | $\text{Var}_t(z_t)$ over steady-state hover. | Context stability. |

---

## 10. Experimental Baselines and Ablation Variants

### 10.1 Comparative Baselines (B1–B6)
* **B1 (PID-Only):** Classical cascaded DSL PID tracking A* global waypoints with demo-guided velocity.
* **B2 (PPO MLP + PID):** High-level PPO MLP policy + fixed DSL PID.
* **B3 (PPO MLP + Adaptive PID):** PPO MLP + Heuristic `AdaptiveGainScheduler` (Phase 0 baseline).
* **B4 (PPO MLP + MPC + PID):** PPO MLP + Fixed Horizon Quadratic MPC ($H=20$) + fixed PID.
* **B5 (PPO MLP + MPC + Adaptive PID):** PPO MLP + Adaptive MPC + Heuristic Adaptive PID.
* **B6 (Transformer PPO + MPC + Adaptive PID / Domain Randomization):** Temporal Transformer PPO trained under Domain Randomization + Adaptive MPC + Adaptive PID.

### 10.2 Ablation Variants (A0–A4)
To scientifically isolate the contribution of each architectural component (satisfying Hypothesis H3):

* **A0 (Static Baseline):** Transformer PPO + MPC + PID with all meta-reconfiguration disabled ($c_t = c_0$).
* **A1 (Context-Driven RL Adaptation Only):** Transformer context $z_t$ only modulates RL blending authority ($\lambda_{\text{RL}}$); MPC and PID parameters remain fixed.
* **A2 (Context + MPC Adaptation Only):** Context $z_t$ modulates $\lambda_{\text{RL}}$ and MPC parameters ($H, Q, R$); PID gains remain static ($K = K_0$).
* **A3 (Context + PID Adaptation Only):** Context $z_t$ modulates $\lambda_{\text{RL}}$ and PID gains ($K_P, K_I, K_D$); MPC parameters remain static.
* **A4 (Full MCR-UAV):** Complete joint multi-tier reconfiguration of $\lambda_{\text{RL}}$, MPC ($H, Q, R$), and PID ($K_P, K_I, K_D$).

---

## 11. Disturbance Task Distributions (Train vs. Held-Out vs. OOD)

To prevent data leakage and ensure rigorous research integrity, tasks are partitioned into three strictly disjoint sets:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │  META-TRAINING DISTRIBUTION (Train Set: 80 Tasks)                      │
  │  - Wind: [0.0, 2.5] m/s, Freq: [0.5, 2.0] Hz                          │
  │  - Impulse: [1.0, 3.0] N                                               │
  │  - Sensor Noise: [1.0x, 2.0x] baseline variance                        │
  │  - Motor Degradation: [0%, 25%] LoE on rotors 1-4                      │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  HELD-OUT META-VALIDATION SET (Val Set: 20 Tasks, Disjoint Seeds)      │
  │  - Interpolation within training ranges with novel waypoint geometry   │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  OUT-OF-DISTRIBUTION (OOD) META-TEST DISTRIBUTION (Test Set: 30 Tasks) │
  │  - Extreme Wind: [3.0, 5.5] m/s (Unseen during training)               │
  │  - Extreme Impulse: [4.0, 6.0] N at random onset t ∈ [1.0s, 4.0s]      │
  │  - Severe Sensor Attack: [3.5x, 5.0x] variance                         │
  │  - Severe Actuator Loss: [30%, 70%] LoE on asymmetric rotor pairs      │
  │  - Compound Disturbances: (Wind + Sensor Attack + Actuator Degradation)│
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Motor Degradation Experimental Sweep

A controlled motor degradation sweep will be executed across all baselines and MCR-UAV:

* **Degradation Levels Evaluated:** $\delta \in \{0\%, 10\%, 20\%, 30\%, 40\%, 50\%, 60\%, 70\%\}$.
* **Fault Injection Mechanism:** Applied to motors 1 and 2 simultaneously (severe asymmetric roll/pitch cross-coupling):
  $$\mathbf{F}_{\text{thrust}, i} = (1 - \delta) \cdot k_f \omega_i^2, \quad i \in \{1, 2\}$$
* **Evaluation Protocol:** 10 evaluation episodes per degradation level per random seed (5 seeds $\times$ 8 levels $\times$ 7 configurations = 2,800 test episodes).

> [!CAUTION]  
> **Claim Boundary:** We do NOT promise survival at any specific percentage (e.g. $70\%$) before experimentation. The maximum survivable degradation threshold $\delta_{\text{max}}$ will be determined strictly from empirical results.

---

## 13. Disturbance Onset and Adaptation Protocol

To measure transient adaptation dynamics and test Hypothesis H4:

1. **Nominal Flight Initialization:** Drone takes off and tracks waypoints under nominal conditions for $t \in [0.0\text{ s}, 2.0\text{ s}]$ (20 RL steps).
2. **Abrupt Fault Injection ($t_{\text{fault}} = 2.0\text{ s}$):** Disturbance/fault is injected instantaneously without environment reset.
3. **Adaptation Metrics Measured:**
   * **Adaptation Steps ($N_{\text{adapt}}$):** Number of decision steps after $t_{\text{fault}}$ until tracking error returns to and remains within $e_{\text{thresh}} = 0.20\text{ m}$.
   * **Adaptation Time ($T_{\text{adapt}}$):** $T_{\text{adapt}} = N_{\text{adapt}} \times \Delta t_{\text{RL}}$.
   * **Recovery Declaration:** Recovery is formally declared if and only if $\|\mathbf{p}_t - \mathbf{p}_t^{\text{ref}}\|_2 \le 0.20\text{ m}$ for $\ge 20$ consecutive control cycles without entering attitude flip ($|\text{roll}|, |\text{pitch}| < 60^\circ$).

---

## 14. Ablation Study Protocol

The ablation study will evaluate configurations **A0, A1, A2, A3, and A4** across:
1. Nominal Tracking (Scenario A)
2. Sustained Wind (Scenario B)
3. High-Frequency Gusts (Scenario C)
4. Sudden Impulse (Scenario D)
5. Sensor Noise Attack (Scenario E)
6. Asymmetric Actuator Degradation (Scenario F, $\delta = 30\%$)

**Ablation Evaluation Matrix:**
* Metrics recorded: Success Rate ($\%$), RMSE ($\text{m}$), Peak Jerk ($\text{m/s}^3$), Energy ($\text{RPM}^2$), and Recovery Time ($T_{\text{rec}}$).

---

## 15. Novelty Threat Validation and Comparative Protocol

To defend against the primary novelty threats identified in the Literature Freshness Audit (Phase 1.5):

| Prior Art / Novelty Threat | Published Focus | Our Architectural Comparative Baseline | Key Hypothesis to Defend |
| :--- | :--- | :--- | :--- |
| **MAVEN (Zhang et al., 2026)** | Monolithic end-to-end Meta-RL with predictive context encoder. | **Comparative Baseline C1 (Monolithic Meta-RL):** Re-implements predictive context encoder outputting body rates directly to rotor mixer without MPC/PID layers. | *Hierarchical multi-tier reconfiguration (MCR-UAV) achieves lower tracking error and higher safety margin than monolithic Meta-RL under trajectory constraints.* |
| **RAPTOR (Bauersfeld et al., 2024/2025)** | Foundation Transformer policy with domain randomization. | **Comparative Baseline B6:** Evaluates our domain-randomized Transformer baseline on severe asymmetric faults. | *Passive in-context recurrence fails ($0.0\%$) under severe asymmetric faults; active meta-reconfiguration is required.* |
| **AC4MPC (Wang & Spenko, 2025)** | Online RL tuning of MPC horizon and weights in isolation. | **Comparative Baseline A2 (Context + MPC Adaptation Only):** Evaluates isolated MPC parameter tuning without low-level PID gain adaptation. | *Isolated MPC tuning cannot damp high-frequency rotor chatter; joint PID gain reconfiguration is essential.* |
| **Neural-Fly (O'Connell et al., 2022)** | Offline aerodynamic basis with $\mathcal{L}_1$ adaptive feedforward wind compensation. | **Comparative Baseline C2 (Adaptive Feedforward Baseline):** Evaluates additive feedforward wrench compensation. | *Additive feedforward wrench compensation is effective for wind but fails when rotor control authority is degraded.* |

---

## 16. Statistical Evaluation Methodology

To guarantee scientific reproducibility and statistical significance:

1. **Evaluation Volume:** $\ge 5$ random seeds for all benchmark experiments, $\ge 10$ evaluation episodes per task condition per seed.
2. **Descriptive Statistics:** Report $\text{Mean} \pm \text{Standard Deviation}$ and $95\%$ Confidence Intervals ($\text{CI}_{95}$) computed via bootstrap resampling ($1,000$ iterations).
3. **Hypothesis Testing:**
   * Paired-sample comparisons between matched seeds/tasks will be evaluated using the **Wilcoxon Signed-Rank Test** (non-parametric) or **Paired t-test** (where normality holds via Shapiro-Wilk test).
   * Significance threshold set to $\alpha = 0.05$.
4. **Effect Size:** Report **Cohen's $d$** to quantify practical significance of performance deltas.
5. **Data Integrity:** No cherry-picked runs; all failed episodes will be recorded in the master results database.

---

## 17. Objective Success Criteria for Hypotheses

| Hypothesis | Quantitative Success Criterion | Calibration Status |
| :--- | :--- | :---: |
| **H1 (OOD Robustness)** | $\text{SR}_{\text{MCR}} - \text{SR}_{\text{B6}} \ge +20\%$ and $\text{RMSE}_{\text{MCR}} < \text{RMSE}_{\text{B6}}$ with $p < 0.05$ on OOD Test Set. | Objective Defined |
| **H2 (Actuator Faults)** | $\text{SR}_{\text{MCR}} \ge 50\%$ at $\delta = 30\%$ LoE where B1–B6 achieve $0.0\%$ success. | Objective Defined |
| **H3 (Multi-Tier Synergy)** | $\text{RMSE}_{\text{A4}} < \min(\text{RMSE}_{\text{A1}}, \text{RMSE}_{\text{A2}}, \text{RMSE}_{\text{A3}})$ with $p < 0.05$ across compound tasks. | Objective Defined |
| **H4 (Rapid Adaptation)** | Mean $N_{\text{adapt}} \le 10$ steps ($1.0\text{ s}$) post-impulse / fault injection across $\ge 80\%$ of recoverable runs. | Objective Defined |
| **H5 (Nominal Integrity)** | Difference in nominal tracking RMSE $|\text{RMSE}_{\text{MCR}} - \text{RMSE}_{\text{B6}}| \le 0.05\text{ m}$ ($p > 0.05$) under Scenario A. | Objective Defined |

---

## 18. Objective Failure and Rejection Criteria

MCR-UAV will be formally declared **unsuccessful or unviable** if any of the following occur during empirical validation:

1. **Zero Degradation Benefit:** MCR-UAV fails to achieve $\text{SR} > 0.0\%$ at $\delta = 30\%$ motor degradation, demonstrating no improvement over baseline B6.
2. **Instability from Reconfiguration:** Reconfiguration parameter shifts induce high-frequency limit cycles, resulting in average jerk exceeding $10\times$ baseline ($\text{Avg Jerk} > 30\text{ m/s}^3$).
3. **Excessive Computational Overhead:** Forward-pass execution latency of $f_\phi + g_\theta$ exceeds $10.0\text{ ms}$ on CPU, violating real-time feasibility for a $10\text{ Hz}$ control loop.
4. **Context Collapse:** The latent vector $z_t$ collapses to a constant embedding across distinct physical disturbance regimes, indicating failure of the Transformer context encoder.
5. **Nominal Performance Degradation:** Nominal tracking error increases by $>25\%$ compared to baseline B6 due to over-reactive meta-reconfiguration.

---

## 19. Threats to Validity

1. **Simulation-to-Reality Gap:** Experiments are conducted in PyBullet (`gym-pybullet-drones`). While PyBullet models aerodynamic drag, rotor downwash, ground effects, and motor dynamics, unmodeled blade flapping and structural flexure in physical flight may alter quantitative thresholds.
2. **Domain Randomization Parameter Ranges:** If training distributions are chosen too narrowly or broadly, baseline B6 may be artificially penalized. We equalize task sampling budgets across all methods.
3. **Multi-Rate Asynchrony:** Slower $10\text{ Hz}$ parameter updates interacting with $1200\text{ Hz}$ PID inner loops can cause transient gain switching bumps. This is mitigated via low-pass gain filtering and anti-windup freezing.
4. **Seed Sensitivity:** RL policies can exhibit high seed variance. We mandate $\ge 5$ seeds for all final reporting.
5. **Computational Constraints:** Evaluation is focused on algorithmic performance in simulation; physical embedded deployment on microcontroller hardware remains future work.

---

## 20. Formal IEEE-Grade Research Claim

> **Standardized Research Claim:**  
> *"In this paper, we investigate the problem of rapid online adaptation in hierarchical autonomous quadrotor control under dynamic aerodynamic disturbances, sensor corruption, and severe actuator degradation. We propose **MCR-UAV**, a meta-contextual reconfiguration framework in which a temporal Transformer encodes recent onboard transition histories to infer latent physical operating conditions in a single forward pass, and a Meta-RL supervisor dynamically reconfigures control authority, mid-level Model Predictive Control parameters, and low-level PID stabilization gains simultaneously. We evaluate the proposed architecture in simulation against non-adaptive, heuristic adaptive, and domain-randomized baselines across interpolation, out-of-distribution, and progressive motor degradation sweeps. Under the evaluated simulation conditions, we demonstrate that coordinated multi-tier reconfiguration achieves faster disturbance recovery and superior fault resilience compared to single-layer adaptation, without requiring online gradient descent or policy retraining."*

---

## 21. Phase 2 Acceptance Checklist

- [x] **Formal Research Problem Defined:** Clear formulation distinguishing Honors baseline from MCR-UAV contribution (Section 1).
- [x] **Research Objectives Stated:** 1 primary objective and 5 secondary objectives defined (Section 2).
- [x] **Research Questions Formulated:** RQ1 (robustness), RQ2 (multi-tier synergy), RQ3 (forward-pass speed) defined (Section 3).
- [x] **Testable Hypotheses Defined:** H1–H5 formulated without claiming proven status (Section 4).
- [x] **Mathematical Notation Defined:** $s_t, a_t, r_t, \Delta s_t, H_t, z_t, c_t$, $f_\phi$, $g_\theta$, and controller effects mathematically specified (Section 5).
- [x] **Design vs. Implemented Equations Distinguished:** Explicitly categorized (Section 5.3).
- [x] **Multi-Rate Architecture Documented:** $10\text{ Hz}$ meta/MPC vs. $1200\text{ Hz}$ PID safe interaction detailed (Section 6).
- [x] **Parameter Bounds Specified:** Ranges and bounding mechanisms defined with uncalibrated parameters marked (Section 7).
- [x] **Independent & Dependent Variables Listed:** Comprehensive variable tables provided (Sections 8 & 9).
- [x] **Baselines and Ablations Formalized:** B1–B6 and A0–A4 fully defined (Section 10).
- [x] **Task Distributions Structured:** Train (80), Val (20), and OOD Test (30) disjoint sets defined (Section 11).
- [x] **Motor Degradation Sweep Designed:** $0\%$ to $70\%$ experimental sweep defined without unverified claims (Section 12).
- [x] **Adaptation Protocol Defined:** $N_{\text{adapt}}, T_{\text{adapt}}$, and formal recovery declaration specified (Section 13).
- [x] **Ablation Protocol Detailed:** Controlled ablation matrix established (Section 14).
- [x] **Novelty Threats Incorporated:** Comparative baselines against MAVEN, RAPTOR, AC4MPC, and Neural-Fly designed (Section 15).
- [x] **Statistical Methodology Established:** $\ge 5$ seeds, CIs, Wilcoxon/t-tests, and effect sizes specified (Section 16).
- [x] **Success & Failure Criteria Defined:** Objective quantitative acceptance and rejection thresholds established (Sections 17 & 18).
- [x] **Threats to Validity Analyzed:** 5 key validity threats discussed (Section 19).
- [x] **Conservative IEEE Claim Stated:** Publication-grade claim formulated (Section 20).
- [x] **No Unverified Claims as Facts:** All future metrics marked as targets (Entire Document).
