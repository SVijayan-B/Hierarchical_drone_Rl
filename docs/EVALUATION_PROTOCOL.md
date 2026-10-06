# MCR-UAV: Formal Meta-Train / Validation / OOD Test Evaluation Protocol

**Document Version:** 1.0.0  
**Project:** Meta-Contextual Reconfiguration of Hierarchical UAV Control (MCR-UAV)  
**Date:** 2026-08-12  
**Implementation Source:** `evaluation/evaluation_protocol.py`  
**Configuration Source:** `configs/evaluation_protocol.yaml`  
**Task Manifest Source:** `configs/meta_tasks_manifest.json`  

---

## 1. Purpose

The objective of this protocol is to define a standardized, mathematically rigorous, and reproducible experimental pipeline for evaluating autonomous UAV control systems across structured environmental distributions. This protocol establishes the evaluation criteria for comparing frozen Honors baseline models (MLP PPO vs. Transformer PPO) and lays the ground rules for future Meta-RL context-adaptation experiments without data leakage.

---

## 2. Dataset and Task Split Architecture

The benchmark distribution comprises exactly $130$ deterministic, seed-locked tasks partitioned into three mutually exclusive splits:

```
                            ┌──────────────────────────────────┐
                            │    MASTER TASK DATASET: N=130    │
                            │  configs/meta_tasks_manifest.json│
                            └─────────────────┬────────────────┘
                                              │
                 ┌────────────────────────────┼────────────────────────────┐
                 ▼                            ▼                            ▼
  ┌────────────────────────────┐┌────────────────────────────┐┌────────────────────────────┐
  │   META-TRAIN (N = 80)      ││   META-VAL (N = 20)        ││   OOD META-TEST (N = 30)   │
  │   Seeds: 1042 → 1121       ││   Seeds: 2042 → 2061       ││   Seeds: 3042 → 3071       │
  │   In-Distribution Training ││   Model / Hyperparam Tuning││   Strictly Frozen Test Set │
  │   Motor LoE: [0%, 25%]     ││   Motor LoE: [0%, 25%]     ││   Motor LoE: [30%, 70%]    │
  │   Wind: [0.0, 2.5] m/s     ││   Wind: [0.0, 2.5] m/s     ││   Wind: [3.0, 5.5] m/s     │
  └────────────────────────────┘└────────────────────────────┘└────────────────────────────┘
```

---

## 3. Strict Data Flow and Leakage Prevention

To guarantee scientific validity, the protocol enforces strict unidirectional data flow:

$$\text{Meta-Training Tasks (80)} \xrightarrow{\text{Gradient Updates}} \text{Meta-Validation Tasks (20)} \xrightarrow{\text{Model Selection}} \text{Freeze Final Model} \xrightarrow{\text{Zero-Shot / Few-Shot}} \text{OOD Meta-Test (30)}$$

### Non-Negotiable Leakage Rules:
1. **OOD Blindness:** The $30$ OOD Meta-Test tasks must **never** be accessed during training, backpropagation, reward tuning, architecture search, or early stopping.
2. **Seed Namespace Disjointness:** $\text{Seeds}_{\text{Train}} \cap \text{Seeds}_{\text{Val}} \cap \text{Seeds}_{\text{OOD}} = \emptyset$.
3. **Physical Range Separation:** Unseen physical extremes (e.g., motor degradation $\delta \ge 30\%$, wind speed $v_w \ge 3.0\text{ m/s}$, sensor noise $k_{\text{noise}} \ge 3.5\times$) appear exclusively in the OOD test set.

---

## 4. Baseline Evaluation Specifications

The protocol evaluates two frozen baseline models preserved from prior Honors research:

| Parameter | Baseline 1: MLP PPO | Baseline 2: Transformer PPO |
| :--- | :--- | :--- |
| **Model Checkpoint** | `results_hierarchical/run_mlp_obs_dist/final_model.zip` | `results_hierarchical/run_trans_obs_dist/final_model.zip` |
| **VecNormalize** | `results_hierarchical/run_mlp_obs_dist/vecnormalize.pkl` | `results_hierarchical/run_trans_obs_dist/vecnormalize.pkl` |
| **History Usage** | `use_history = False` | `use_history = True` |
| **Observation Dim** | $d_s = 25$ | History buffer: $20 \times 34 = 680$ |
| **Mid-Level MPC** | Enabled ($10\text{ Hz}$, $H=20$) | Enabled ($10\text{ Hz}$, $H=20$) |
| **Low-Level PID** | Adaptive DSL PID ($1200\text{ Hz}$) | Adaptive DSL PID ($1200\text{ Hz}$) |
| **Integrity Rule** | Read-Only Frozen Snapshot | Read-Only Frozen Snapshot |

---

## 5. Adaptation Budget Protocol (For Future Meta-RL Phases)

For future Meta-RL adaptation evaluation (Phases 15–25), the protocol formalizes adaptation budgets:

$$N_{\text{adapt}} \in \{0, 1, 5, 10, 20\} \quad \text{interaction steps (at 10 Hz)}$$

* **0-Shot Adaptation ($N_{\text{adapt}} = 0$):** Immediate forward-pass execution without prior history ($H_0 = \mathbf{0}$).
* **Few-Step In-Context Adaptation ($N_{\text{adapt}} \in \{1, 5, 10, 20\}$):** The context encoder accumulates $N_{\text{adapt}}$ transitions in its history buffer to estimate $z_t = f_\phi(H_t)$ before latching the reconfiguration vector $c_t = g_\theta(z_t)$.

> [!NOTE]
> Phase 5 evaluates zero-adaptation baselines ($N_{\text{adapt}} = 0$). No Meta-RL training or online adaptation is performed in Phase 5.

---

## 6. Performance and Adaptation Metrics

Every evaluation episode records the following comprehensive metric set:

### Primary Flight Metrics
1. **Success Rate:** Binary indicator ($1$ if $\|p - p^{\text{target}}\| \le 0.60\text{ m}$ maintained for $\ge 3.0\text{ s}$, $0$ otherwise).
2. **Tracking RMSE ($\text{m}$):** $\text{RMSE} = \sqrt{\frac{1}{K}\sum_{k=1}^K \|p_k - p^{\text{target}}\|^2}$.
3. **Mean Tracking Error ($\text{m}$):** $\bar{e} = \frac{1}{K}\sum_{k=1}^K \|p_k - p^{\text{target}}\|$.
4. **Max Position Error ($\text{m}$):** $e_{\max} = \max_k \|p_k - p^{\text{target}}\|$.
5. **Settling Time ($T_s$, $\text{s}$):** Time required to enter and remain within the $0.60\text{ m}$ tolerance sphere.
6. **Overshoot ($\text{m}$ & $\%$):** Maximum distance exceeded past the target position along the flight path.

### Smoothness and Control Effort Metrics
7. **Peak Jerk ($\text{m/s}^3$):** $J_{\max} = \max_k \|\dddot{p}_k\|$.
8. **Average Jerk ($\text{m/s}^3$):** $\bar{J} = \frac{1}{K}\sum_{k=1}^K \|\dddot{p}_k\|$.
9. **Control Effort:** $\bar{u}_{\text{effort}} = \frac{1}{K}\sum_{k=1}^K \|v_k^{\text{cmd}}\|^2$.
10. **Energy Proxy:** $E_{\text{total}} = \sum_{k=1}^K \sum_{i=1}^4 \left(\frac{\text{RPM}_{i,k}}{10000}\right)^2$.

### Rapid Adaptation Metrics (Future Phases)
11. **Performance Delta ($\Delta P$):** $\Delta P = P_{\text{post-adapt}} - P_{\text{pre-adapt}}$.
12. **Adaptation Efficiency:** $\eta_{\text{adapt}} = \frac{\Delta P}{N_{\text{adapt}}}$.
13. **Recovery Time ($T_{\text{adapt}}$, $\text{s}$):** Time elapsed between disturbance onset and error return to nominal band ($\le 0.20\text{ m}$).

---

## 7. Statistical Aggregation and Significance Protocol

To guarantee publishable IEEE-level rigor:
1. **Paired Task-Level Evaluation:** Both models are evaluated on identical task instances with identical random seeds.
2. **Aggregated Summaries:** Mean and sample standard deviation reported across all tasks within each split:
   $$\bar{\mu} = \frac{1}{N}\sum_{i=1}^N x_i, \quad \sigma = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (x_i - \bar{\mu})^2}$$
3. **Statistical Significance Testing (Future Phases):**
   * Wilcoxon signed-rank test for paired non-parametric metric distributions ($\alpha = 0.05$).
   * Paired Student's $t$-test for normally distributed tracking errors.
   * Bootstrap $95\%$ confidence intervals ($B = 10,000$ resamples).
   * Cohen's $d$ effect sizes ($d \ge 0.8$ denotes large effect).

---

## 8. Implementation Status Summary

* **Currently Implemented (Phase 5):**
  - Protocol configuration: `configs/evaluation_protocol.yaml`.
  - Protocol executor: `evaluation/evaluation_protocol.py`.
  - Baseline evaluation on 130 tasks: `results/phase5_baseline_evaluation.csv`.
  - Statistical summary table: `results/phase5_baseline_summary.csv`.
* **Planned for Future Phases (Phases 15–25):**
  - Meta-RL Context Encoder forward pass and adaptation budget sweeps.
  - Multi-tier reconfiguration parameter tracking ($c_t = [\lambda_{\text{RL}}, \alpha_Q, \alpha_R, \alpha_P, \alpha_I, \alpha_D, H]$).
  - Hypothesis testing (H1–H5) and Wilcoxon significance tables.
