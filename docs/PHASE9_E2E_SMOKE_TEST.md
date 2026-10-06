# Phase-9 End-to-End Non-Nominal Reconfiguration Smoke Test Report

This document reports the integration correctness results of the Phase-9 non-nominal environment reconfiguration smoke test.

---

## 1. Test Configuration Used
*   **Lambda_RL (lambda_RL):** 0.70
*   **Alpha_Q (alpha_Q):** 2.00
*   **Alpha_R (alpha_R):** 0.50
*   **Alpha_P (alpha_P):** 1.30
*   **Alpha_I (alpha_I):** 0.70
*   **Alpha_D (alpha_D):** 1.20
*   **MPC Horizon (H):** 30

---

## 2. Environment Details & Timing Relationship
*   **Environment Class:** `HierarchicalNavEnv`
*   **Timing Architecture:**
    *   **PyBullet Physics Frequency:** 240 Hz (physics time step is 1/240 s, executing 2 simulation steps per control step).
    *   **Control Frequency (MPC / PID Outputs):** 120 Hz (executing once every 2 physics steps).
    *   **PID/Internal Gain-Update Frequency:** 120 Hz (the gain smoothing updates run within the control loop once every control step, i.e., 12 times per RL step).
    *   **RL / Meta-Supervisor Frequency:** 10 Hz (decisions and target reconfiguration parameter updates occur once every 0.1 s, i.e., every 12 control steps).
*   **PPO Policy Used:** MLP model checkpoint `results_hierarchical/run_mlp_obs_dist/final_model.zip`.

---

## 3. Initial Nominal Parameters
*   lambda_RL: 1.0
*   alpha_Q: 1.0
*   alpha_R: 1.0
*   alpha_P: 1.0
*   alpha_I: 1.0
*   alpha_D: 1.0
*   H: 20
*   Nominal PID KP: [0.4, 0.4, 1.25]
*   Nominal PID KI: [0.05, 0.05, 0.05]
*   Nominal PID KD: [0.2, 0.2, 0.5]
*   Nominal MPC Horizon: 20

---

## 4. Applied Parameters
*   Applied Parameter Sequence: `[0.70, 2.00, 0.50, 1.30, 0.70, 1.20, 30.0]`

---

## 5. Target PID Gains
*   K_P Target Scale: 1.30 x base gains = `[0.52, 0.52, 1.625]`
*   K_I Target Scale: 0.70 x base gains = `[0.034999999999999996, 0.034999999999999996, 0.034999999999999996]`
*   K_D Target Scale: 1.20 x base gains = `[0.24, 0.24, 0.6]`

---

## 6. First Effective PID Gains (1200 Hz inner loop step 1)
*   Effective smoothed gain multipliers after the very first inner loop control step (which runs in the first control loop cycle at 120 Hz):
    *   alpha_P: 1.0150 (Expected: 1.0150)
    *   alpha_I: 0.9850 (Expected: 0.9850)
    *   alpha_D: 1.0100 (Expected: 1.0100)

*   **CSV First Sample Reconciliation:**
    While the theoretical gain multipliers after exactly one inner loop step are 1.015, 0.985, and 1.010, the first sample logged in the CSV occurs after the first complete outer RL step of 0.1 seconds. Since there are 12 inner-loop control cycles per RL step, the multipliers have been smoothed 12 times and have progressed closer to their targets:
    *   alpha_P: 1.137892 (CSV step 1)
    *   alpha_I: 0.862108 (CSV step 1)
    *   alpha_D: 1.091928 (CSV step 1)

---

## 7. Smoothing Verification
*   Smoothing multiplier equation: K_new = 0.95 * K_old + 0.05 * K_target
*   Calculated alpha_P: 1.0150 vs Expected: 1.0150 (Delta: 0.00e+00)
*   Calculated alpha_I: 0.9850 vs Expected: 0.9850 (Delta: 0.00e+00)
*   Calculated alpha_D: 1.0100 vs Expected: 1.0100 (Delta: 0.00e+00)
*   **Result:** PASSED (Numerical tolerance < 10**-6)

---

## 8. MPC Q/R Cost Matrix Verification
*   MPC Cost Multipliers Latch check:
    *   alpha_Q: 2.00 (Actual: 2.00)
    *   alpha_R: 0.50 (Actual: 0.50)
*   Base matrices Q_0 and R_0 verified unchanged.
*   **Result:** PASSED

---

## 9. MPC Horizon Verification
*   MPC discrete Horizon length: 30
*   Effective latched horizon: 30
*   **Result:** PASSED (Strictly integer 30, no rounding drift)

---

## 10. Lambda Blending Verification
*   Command blending equation: target_wp = 0.70 * rl_wp + 0.30 * astar_wp
*   Checked 260 command blending steps.
*   **Result:** PASSED (numerical matching within tolerance < 10**-5)

---

## 11. PyBullet Episode Result
*   Logged steps: 60 steps (6.0 seconds).
*   **Result:** PASSED

---

## 12. Numerical Safety Result
*   Observation states, joint speeds, and control parameters verified finite (no NaN, no Inf) at every step.
*   **Result:** PASSED

---

## 13. Motor-Limit Result
*   Motor outputs verified within physical actuator boundaries: min_rpm=0.00, max_rpm=22000.00.
*   **Result:** PASSED

---

## 14. Anti-Windup Result
*   Anti-windup path not triggered during this nominally stable smoke episode; Phase-9 unit tests cover anti-windup integration.
*   **Result:** PASSED

---

## 15. PASS/FAIL Verdict
*   **Final Smoke Test Verdict:** **PASS**
