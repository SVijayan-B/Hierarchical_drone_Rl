# Hierarchical Autonomous Drone System (RL + Cascaded PID)

This project implements a **hierarchical autonomous drone control stack** in simulation using:

- `gym-pybullet-drones`
- `PyBullet`
- `Stable-Baselines3 PPO`
- IMU + ultrasonic sensing (no camera)

The architecture follows a realistic cascaded concept:

`RL policy (high-level) -> velocity command -> PID stabilization (low-level) -> motor RPM -> drone physics`

---

## 1. Core Idea

The drone is controlled by **two layers**:

1. **High-level decision layer (RL/PPO)**
- Decides navigation intent (velocity direction/magnitude).
- Learns to move toward target while staying safe/stable from reward feedback.

2. **Low-level stabilization layer (PID)**
- Converts high-level commands into physically stable actuation.
- Maintains roll/pitch/yaw and thrust behavior robustly.
- In this repo, low-level stabilization is performed with `gym_pybullet_drones.control.DSLPIDControl`.

This separation is important:
- RL handles strategy.
- PID handles fast stabilization.

---

## 2. Project Structure

```text
hierarchical_drone/
  config/
    settings.py
  env/
    hierarchical_nav_env.py
  sensors/
    imu.py
    ultrasonic.py
  controllers/
    pid_controller.py
    velocity_controller.py
    attitude_controller.py
  rl/
    train_ppo.py
    evaluate.py
  main.py
```

### What each part does

- `config/settings.py`
  - All tunable parameters: frequencies, velocity limits, reward thresholds, wind disturbance values, etc.

- `env/hierarchical_nav_env.py`
  - Main Gymnasium environment.
  - Spawns drone + target, applies wind disturbance, computes observations, reward, and done logic.
  - Integrates RL commands with PID stabilization.

- `sensors/imu.py`
  - Simulated IMU-like outputs from drone state (angles/rates/acceleration + optional noise).

- `sensors/ultrasonic.py`
  - Simulated ultrasonic distances via PyBullet ray tests (front/left/right/rear/down).

- `controllers/*.py`
  - Contains reusable PID/cascade components.
  - Current env uses `DSLPIDControl` from `gym-pybullet-drones` for low-level stabilization.

- `rl/train_ppo.py`
  - PPO training pipeline with `VecNormalize`, checkpointing, evaluation callback, TensorBoard logging.

- `rl/evaluate.py`
  - Deterministic policy evaluation with GUI support and terminal reporting (reward + HIT/MISS).

- `main.py`
  - Entry point for train/eval modes.

---

## 3. Cascaded PID Concept (Control Side)

Even when RL provides guidance, physical flight stability requires fast inner loops.

Conceptually, cascaded control is:

1. **Outer loop**: velocity/position objective
2. **Middle loop**: attitude objective (roll/pitch/yaw)
3. **Inner loop**: rate/torque stabilization
4. **Mixer**: convert torque/thrust to motor RPM

In this project:
- High-level command is interpreted as desired velocity intent.
- `DSLPIDControl` handles conversion into motor-level control robustly.
- This keeps drone stable under disturbances and noise better than end-to-end motor RL.

Why this is better than RL->RPM directly:
- Better stability
- Faster convergence
- More realistic autopilot structure
- Easier sim-to-real transition

---

## 4. RL Observation (State) Design

The observation vector in `hierarchical_nav_env.py` contains normalized values of:

- Drone pose and motion:
  - position `(x,y,z)`
  - linear velocity `(vx,vy,vz)`
  - attitude `(roll,pitch,yaw)`
  - angular rates `(p,q,r)`

- Goal relation:
  - relative target vector `(dx,dy,dz)`
  - distance to target

- Ultrasonic sensing:
  - front, left, right, rear, down ranges

- Control history:
  - previous action (4D)

Why this matters:
- Position/velocity tells “where and how fast”.
- IMU terms tell “how stable or tilted”.
- Ultrasonic terms give local safety.
- Relative target terms give navigation objective.
- Previous action helps command smoothness.

---

## 5. RL Action Design

Action space is 4D (normalized `[-1,1]`):

- `a0`: desired x-velocity intent
- `a1`: desired y-velocity intent
- `a2`: desired z-velocity intent
- `a3`: desired yaw-rate intent

Then action is:
- clipped
- rate-limited
- low-pass smoothed

This avoids twitchy/aggressive commands.

> Note: In `demo_guided_mode=True`, policy action is intentionally overridden for deterministic stable demos.

---

## 6. Reward Function (Shaped)

Reward combines progress, stability, safety, and smoothness.

### Positive terms
- `progress_reward`: reward for reducing XY target distance each step
- `target_bonus`: extra reward when entering target vicinity

### Negative terms
- obstacle proximity penalty (from ultrasonic)
- tilt penalty (roll/pitch magnitude)
- angular-rate penalty
- action smoothness penalty (large command jumps)
- effort penalty (very high velocity command)
- velocity stability penalty (high translational jitter)
- collision/out-of-bounds penalties

This reward shaping encourages:
- moving toward the target
- flying stably (low oscillation)
- safe and smooth behavior

---

## 7. Success / Termination Logic

Current success is stricter than “touching target”:

- drone must be near target XY
- maintain stable altitude/attitude/velocity
- remain stable over target for ~3 seconds continuously

Other termination conditions:
- out of bounds
- max episode length
- fall/collision after grace window

This enforces true **hover-on-target stability**.

---

## 8. Wind Disturbance Model

Configured in `SimConfig`:

- `wind_disturbance = (fx, fy, fz)` amplitudes in Newtons
- `wind_freq_hz = (wx, wy, wz)` sinusoidal frequencies
- `wind_gust_scale` random gust factor

In each control step, environment applies:
- oscillatory wind force
- plus random gust noise

This tests disturbance rejection and makes roll/pitch/yaw stabilization meaningful.

---

## 9. PPO Training Pipeline

Training file: `rl/train_ppo.py`

Flow:
1. Build environment
2. Wrap with `Monitor`
3. Wrap with `VecNormalize` (obs + reward normalization)
4. Train PPO (`MlpPolicy`)
5. Use callbacks:
   - checkpoints
   - periodic eval
   - best model save
6. Save final model + vecnormalize stats

Why normalization is important:
- Observation scales vary (angles, velocities, distances).
- Normalization stabilizes PPO optimization.

---

## 10. Evaluation Pipeline

Evaluation file: `rl/evaluate.py`

- Loads model and `VecNormalize` statistics
- Runs deterministic rollout in GUI
- Prints per-episode:
  - reward
  - target status (`HIT`/`MISS`)

---

## 11. How to Run

### Train
```powershell
cd D:\IT SECTOR\PROJECTS\Drone3\RL-Drone
python -m hierarchical_drone.main --mode train --timesteps 500000
```

### Evaluate latest run
```powershell
$run = Get-ChildItem .\results_hierarchical | Sort-Object LastWriteTime -Descending | Select-Object -First 1
python -m hierarchical_drone.main --mode eval --model_path "$($run.FullName)\final_model.zip" --vecnorm_path "$($run.FullName)\vecnormalize.pkl" --episodes 1
```

If model is saved without `.zip`, use `final_model` path.

---

## 12. Tuning Tips

If unstable:
- lower `ActionConfig.vxy_max`, `vz_max`
- increase smoothing (`action_smoothing_alpha` lower updates per step with stronger LPF behavior)
- reduce wind amplitudes
- increase `target_threshold_m`

If too slow:
- increase `vxy_max` gradually
- reduce heavy penalties slightly

If no learning:
- ensure `demo_guided_mode=False` for true policy learning behavior
- retrain after major reward/logic changes

---

## 13. Important Note on Demo vs Learning Modes

- `demo_guided_mode=True`:
  - deterministic guided behavior for reliable visual demos.
  - PPO action is ignored.

- `demo_guided_mode=False`:
  - PPO action actually drives high-level command.
  - required for genuine RL performance validation.

Use demo mode for presentation stability, and policy mode for research/training validation.
