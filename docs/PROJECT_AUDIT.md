# MCR-UAV: Comprehensive Repository & Architecture Audit (Phase 0)

**Document Version:** 1.0.0  
**Project:** Meta-Contextual Reconfiguration of Hierarchical UAV Control (MCR-UAV)  
**Date:** 2026-08-12  
**Baseline Status:** Frozen in `baseline_honors/` and verified.

---

## 1. Existing System Architecture

The existing repository implements a hierarchical quadrotor autonomous navigation and control stack in PyBullet (`gym-pybullet-drones`). The control flow operates across three hierarchical tiers:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 HIGH-LEVEL DECISION TIER               │
                  │   Transformer Feature Extractor / PPO Policy (10 Hz)   │
                  │    - Input: Stacked Observation History (20 x obs_dim) │
                  │    - Output: 3D Waypoint Velocity Target (vx, vy, vz)  │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │                 MID-LEVEL CONTROL TIER                 │
                  │              Adaptive MPC Controller (10 Hz)           │
                  │    - Dynamic Horizons (H ∈ {10, 20, 30})               │
                  │    - Adaptive Cost Matrices Q_t, R_t                   │
                  │    - Output: Desired Smooth Velocity Vector            │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │                LOW-LEVEL STABILIZATION TIER            │
                  │       Velocity PID & Upgraded Attitude PID (1200 Hz)   │
                  │    - Velocity PID: Desired Thrust & Target Euler       │
                  │    - Upgraded Attitude PID: Anti-windup, D-filter      │
                  │    - Adaptive Gain Scheduler: Heuristic scaling        │
                  │    - Output: Individual Motor RPMs (4 rotors)          │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │   Quadrotor Simulation  │
                                 │  Crazyflie 2.X Dynamics │
                                 └─────────────────────────┘
```

---

## 2. Existing Repository Modules & File Map

| Module / Path | Type | Primary Role & Description |
| :--- | :--- | :--- |
| `pid_vs_rl_pid.py` | Top-level CLI Entry | Evaluates 6-way comparison across tracking, settling time, energy, and honors metrics. |
| `disturbance_recovery_benchmark.py` | Top-level CLI Entry | Evaluates 6 controller configurations across 6 distinct disturbance scenarios (A–F). |
| `hierarchical_drone/main.py` | Demo / Visualizer | Interactive demonstration script for visual flight evaluation. |
| `hierarchical_drone/config/settings.py` | Configuration | Dataclasses for `SimConfig`, `TaskConfig`, `SensorConfig`, `ActionConfig`, `NoiseConfig`. |
| `hierarchical_drone/env/hierarchical_nav_env.py` | Gym Environment | Core environment wrapping `CtrlAviary`, A* global planner, obstacle generator, wind/noise simulation. |
| `hierarchical_drone/controllers/attitude_controller.py` | Controller | Standalone attitude PID implementation. |
| `hierarchical_drone/controllers/velocity_controller.py` | Controller | Standalone velocity PID implementation. |
| `hierarchical_drone/controllers/pid_controller.py` | Controller | `UpgradedDSLPIDControl` subclassing `gym-pybullet-drones` DSL PID with tilt limits, anti-windup, and derivative filtering. |
| `hierarchical_drone/controllers/mpc_controller.py` | Controller | Discrete-time quadratic MPC with dynamic horizons (`H=10, 20, 30`) and velocity filtering. |
| `hierarchical_drone/controllers/adaptive_scheduler.py` | Controller | Heuristic `AdaptiveGainScheduler` computing gain scale factors based on velocity, rotation rate, distance, and wind. |
| `hierarchical_drone/controllers/rl_gain_scheduler.py` | Controller | Legacy RL neural gain scheduler (deprecated/frozen in Phase 6). |
| `hierarchical_drone/rl/transformer_feature_extractor.py`| Neural Network | PyTorch custom SB3 features extractor over 20-step stacked observation history. |
| `hierarchical_drone/rl/train_ppo.py` | Training Pipeline | SB3 PPO training script supporting MLP/Transformer policies, curriculum learning, and vecnormalization. |
| `hierarchical_drone/rl/evaluate.py` | Evaluation | Policy rollout and evaluation utility. |
| `hierarchical_drone/sensors/imu.py` | Sensor Model | 6-DOF IMU sensor model with Gaussian noise injection on angles, rates, and accelerations. |
| `hierarchical_drone/sensors/ultrasonic.py` | Sensor Model | 5-beam ultrasonic distance sensor model (front, left, right, rear, down). |
| `hierarchical_drone/utils/a_star.py` | Global Planner | 2D/3D grid-based A* path planner for obstacle avoidance waypoints. |
| `hierarchical_drone/utils/telemetry_logger.py` | Logging & Telemetry| Comprehensive CSV and NPZ telemetry logger recording physical state, control signals, and rewards. |
| `hierarchical_drone/utils/math_utils.py` | Utilities | Quaternion, Euler, and geometry helper routines. |
| `hierarchical_drone/benchmark/compare_runner.py` | Benchmark Engine | Benchmark test runner executing matched-seed evaluations across the 6 configurations. |
| `hierarchical_drone/benchmark/metrics.py` | Metrics | Performance metrics computation (tracking error, jerk, control effort, oscillation index, sim-to-real readiness). |

---

## 3. Dependencies & Software Environment

* **Python Version:** 3.10+ (Active conda environment: `drones`)
* **Core Frameworks:**
  * `torch == 2.11.0`
  * `stable-baselines3 == 2.8.0`
  * `gymnasium == 1.2.3`
  * `pybullet == 3.2.7`
  * `gym-pybullet-drones == 1.0.0` (commit `bc25d1101321691cf4e5166565509e111dc5489c`)
* **Scientific Computing & Visualization:**
  * `numpy == 2.2.6`
  * `scipy == 1.15.3`
  * `pandas == 2.3.3`
  * `matplotlib == 3.10.8`
  * `tensorboard == 2.20.0`
  * `transforms3d == 0.4.2`

---

## 4. Current Six Controller Configurations

The existing Honors framework establishes 6 comparative configurations evaluated with matched seeds:

1. **`PID-Only`**: Baseline DSL PID controller tracking A* waypoints directly with demo-guided velocity commands.
2. **`PPO MLP + PID`**: PPO policy with a Multi-Layer Perceptron (MLP) generating velocity targets, stabilized by baseline DSL PID.
3. **`PPO MLP + Adaptive PID`**: PPO MLP velocity policy + heuristic `AdaptiveGainScheduler` dynamically scaling PID gains.
4. **`PPO MLP + MPC + PID`**: PPO MLP velocity policy + Mid-level MPC optimization layer (fixed horizon $H=20$) + baseline PID.
5. **`PPO MLP + MPC + Adaptive PID`**: PPO MLP + Adaptive MPC (dynamic horizons $H \in \{10, 20, 30\}$ and adaptive $Q/R$ weights) + Adaptive PID.
6. **`Transformer PPO + MPC + Adaptive PID` (`Trans+M+A`)**: Temporal Transformer Feature Extractor encoding 20-step history + Adaptive MPC + Adaptive PID.

---

## 5. Current Six Disturbance Scenarios

Implemented in `disturbance_recovery_benchmark.py`:

* **Scenario A — No Wind:** Nominal baseline conditions with static obstacle field.
* **Scenario B — Constant Wind:** Constant external drag force vector $\vec{F}_{\text{wind}} = [2.0, 0.0, 0.0] \times 0.015\text{ N}$.
* **Scenario C — Random Gusts:** Sinusoidal time-varying wind force with $2.0\text{ Hz}$ frequency plus stochastic Gaussian turbulence.
* **Scenario D — Impulse Disturbance:** Severe lateral force pulse ($4.5\text{ N}$ scaled) injected at step $t=20$ ($2.0\text{s}$) to evaluate impulse recovery time.
* **Scenario E — Sensor Noise Attack:** High-amplitude sensor corruption ($3\times$ standard IMU and ultrasonic noise).
* **Scenario F — Motor Degradation:** Asymmetric actuator loss ($30\%$ thrust loss on motors 1 and 2: $\eta = [0.70, 0.70, 1.0, 1.0]$).

---

## 6. Current Benchmark Metrics

Computed in `hierarchical_drone/benchmark/metrics.py` and exported by `telemetry_logger.py`:

* **Success Rate ($\%$):** Percentage of episodes reaching the target within sphere radius $0.20\text{ m}$ and maintaining hover for $2.0\text{ s}$.
* **Mean & RMSE Tracking Error ($\text{m}$):** $\frac{1}{T}\sum_{t=1}^T \|\mathbf{p}_t - \mathbf{p}_t^{\text{ref}}\|_2$ and $\sqrt{\frac{1}{T}\sum_{t=1}^T \|\mathbf{p}_t - \mathbf{p}_t^{\text{ref}}\|_2^2}$.
* **Peak & Average Jerk ($\text{m/s}^3$):** Third derivative of position indicating flight smoothness and actuator chatter.
* **Total Energy Proxy ($\times 10^4\text{ RPM}^2$):** Summed squared rotor speeds $\sum_{t=1}^T \sum_{i=1}^4 \omega_{i,t}^2$.
* **Energy per Meter ($\text{RPM}^2/\text{m}$):** Total energy divided by trajectory path length.
* **Settling Time ($\text{s}$):** Time required to enter and remain within target threshold.
* **Trajectory Smoothness Index:** Integral of squared accelerations.
* **Control Effort:** Normalized actuator variation metric.
* **Impulse Recovery Time ($\text{s}$):** Time required to return to within $0.15\text{ m}$ of reference trajectory following impulse perturbation.

---

## 7. Known Baseline Limitations & Identified Research Gaps

1. **Actuator Degradation Failure:** All existing baseline controllers (PID, MLP, and Transformer) achieve **$0.0\%$ success rate** under Scenario F (asymmetric motor degradation), because none dynamically reallocate control authority or reconfigure mixer/controller limits.
2. **Fixed Controller Weights:** The Transformer PPO policy is trained under static domain randomization and cannot dynamically reconfigure its mid-level (MPC) or low-level (PID) authority during runtime interaction.
3. **Absence of Latent Context Inference:** The current temporal Transformer extracts spatial features for policy rollouts, but does not infer a latent task representation $z_t$ capturing unobserved physical parameters (mass, wind, degradation).
4. **Lack of Meta-Supervisory Adaptation:** Current adaptation is limited to hardcoded heuristic rules (`AdaptiveGainScheduler`) rather than meta-learned reconfiguration.

---

## 8. Files Frozen & Protected from In-Place Overwriting

The following baseline implementations are frozen and backed up in `baseline_honors/`:
* `hierarchical_drone/controllers/pid_controller.py`
* `hierarchical_drone/controllers/mpc_controller.py`
* `hierarchical_drone/controllers/adaptive_scheduler.py`
* `hierarchical_drone/env/hierarchical_nav_env.py`
* `hierarchical_drone/rl/transformer_feature_extractor.py`
* `pid_vs_rl_pid.py`
* `disturbance_recovery_benchmark.py`

---

## 9. Recommended Integration Points for Meta-RL (MCR-UAV)

1. **Context Data Pipeline (`models/context_history.py`):** Tap environment transition tuples $(s_t, a_t, r_t, \Delta s_t)$ into a dedicated sequence buffer.
2. **Transformer Context Encoder (`models/transformer_context_encoder.py`):** Map interaction history $H_t$ to latent context vector $z_t \in \mathbb{R}^{d_z}$.
3. **Meta-RL Supervisor (`models/meta_reconfiguration_policy.py`):** Map $z_t \to c_t = [\lambda_{\text{RL}}, \alpha_Q, \alpha_R, \alpha_P, \alpha_I, \alpha_D, H_{\text{MPC}}]$.
4. **Meta-Adaptive MPC (`controllers/meta_adaptive_mpc.py`):** Reconfigure horizon $H(z_t)$ and cost scalers $Q(z_t), R(z_t)$.
5. **Meta-Adaptive PID (`controllers/meta_adaptive_pid.py`):** Reconfigure PID gains $K_P(z_t), K_I(z_t), K_D(z_t)$ with rigorous stability bounds.
6. **Unified MCR-UAV Controller (`controllers/mcr_uav_controller.py`):** Harmonize high-level Transformer-PPO, Meta-Supervisor, MPC, and PID.
