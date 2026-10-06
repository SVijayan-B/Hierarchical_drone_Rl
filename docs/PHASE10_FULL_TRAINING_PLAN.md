# Phase 10 Full Meta-RL Training Plan

This document outlines the parameters and settings for the full multi-seed Meta-RL training experiment.

---

## 1. Task Distribution
*   **Meta-Train Split:** 80 tasks (Seeds 1042 to 1121).
*   **Meta-Validation Split:** 20 tasks (Seeds 2042 to 2061).
*   **OOD Meta-Test Split:** 30 tasks (Seeds 3042 to 3071) [STRICTLY LOCKED].

---

## 2. Optimization Settings
*   **Algorithm:** REINFORCE with Categorical Horizon Sampling.
*   **Trainable Modules:** Context Encoder (104,528 params) and Meta-Supervisor (5,961 params).
*   **Learning Rate:** 0.001
*   **Discount Factor Gamma:** 0.99
*   **Exploration Std:** 0.05
*   **Optimizer:** Adam
*   **Gradient Clipping Norm:** 1.0

---

## 3. Training & Checkpoint Schedule
*   **Episodes per task per iteration:** 5
*   **Total Iterations:** 50
*   **Validation Interval:** Every 5 iterations.
*   **Checkpoint Save Interval:** Every 5 iterations.
*   **Early Stopping:** If validation success rate reaches 100% or loss stabilizes for 10 iterations.

---

## 4. Multi-Seed Replication
To guarantee statistical significance, training will be executed across 5 independent seeds:
*   Seed 42
*   Seed 43
*   Seed 44
*   Seed 45
*   Seed 46

Checkpoints will be saved in separate subdirectories under `results/meta_rl/seed_<seed>/`.
