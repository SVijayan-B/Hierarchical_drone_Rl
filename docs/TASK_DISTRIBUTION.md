# MCR-UAV: Formal Meta-RL Task Distribution and Generation Protocol

**Document Version:** 1.0.0  
**Project:** Meta-Contextual Reconfiguration of Hierarchical UAV Control (MCR-UAV)  
**Date:** 2026-08-12  
**Implementation Source:** `environments/task_generator.py`  
**Configuration Source:** `configs/meta_tasks.yaml`  

---

## 1. Why Meta-RL Requires Structured Task Distributions

Conventional reinforcement learning trains a policy on a single fixed Markov Decision Process (MDP) or across a monolithic domain-randomized parameter set. In contrast, **Meta-Reinforcement Learning (Meta-RL)** assumes the environment is drawn from an underlying distribution of MDPs $\mathcal{M}_i \sim p(\mathcal{T})$, where each task $\mathcal{T}_i$ possesses unique, unobserved transition dynamics $T_{\mathcal{T}_i}(s' \mid s, a)$ and reward functions $R_{\mathcal{T}_i}(s, a)$.

For autonomous UAV control, formalizing a structured task distribution is essential because:
1. **Explicit In-Distribution vs. Out-of-Distribution (OOD) Separation:** It prevents empirical over-fitting and false claims of generalization by maintaining strict statistical boundaries between training tasks and unseen evaluation tasks.
2. **Context Inference Validation:** It allows the temporal Transformer Context Encoder ($f_\phi$) to be trained across diverse dynamic modes so that the latent context $z_t$ captures actionable physical factors (e.g., wind drag, rotor thrust loss, mass shift).
3. **Reproducibility:** A deterministic, seedable task generator guarantees that every baseline algorithm and ablation is evaluated against the exact same physical disturbances with identical onset times and magnitudes.

---

## 2. Distinction: Benchmark Scenarios vs. Meta-RL Task Distributions

* **Benchmark Scenarios (Phase 0 Baseline):** Fixed, coarse-grained evaluation modes (Scenario A: No Wind, Scenario B: Constant Wind, Scenario C: Random Gusts, Scenario D: Impulse, Scenario E: Sensor Attack, Scenario F: Motor Degradation) designed for quick qualitative smoke tests.
* **Meta-RL Task Distributions (Phase 3 Framework):** A structured continuum of $130$ precisely parameterized, reproducible tasks with randomized physical parameters (wind vectors, gust frequencies, turbulence standard deviations, impulse timings, noise scalings, asymmetric rotor degradation percentages, and mass/inertia offsets) sampled from defined probabilistic distributions.

---

## 3. Parameter Ranges Across Task Splits

| Task Parameter | Symbol | Unit | Meta-Training Range (80 Tasks) | Meta-Validation Range (20 Tasks) | OOD Meta-Test Range (30 Tasks) | Separation Rule |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Wind Magnitude** | $v_w$ | $\text{m/s}$ | $[0.0, 2.5]$ | $[0.0, 2.5]$ | $[3.0, 5.5]$ | OOD strictly $> 2.5\text{ m/s}$ |
| **Wind Direction** | $\hat{\mathbf{w}}$ | unit vec | Uniform on horizontal plane | Uniform on horizontal plane | Uniform on horizontal plane | Independent sampling |
| **Gust Magnitude** | $v_g$ | $\text{m/s}$ | $[0.0, 1.5]$ | $[0.0, 1.5]$ | $[2.0, 4.0]$ | OOD strictly $> 1.5\text{ m/s}$ |
| **Gust Frequency** | $f_g$ | $\text{Hz}$ | $[0.5, 2.0]$ | $[0.5, 2.0]$ | $[2.5, 5.0]$ | OOD strictly $> 2.0\text{ Hz}$ |
| **Turbulence Std** | $\sigma_{\text{turb}}$ | $\text{m/s}$ | $[0.0, 0.15]$ | $[0.01, 0.15]$ | $[0.20, 0.45]$ | OOD strictly $> 0.15\text{ m/s}$ |
| **Impulse Force** | $F_{\text{imp}}$ | $\text{N}$ | $[1.0, 3.0]$ | $[1.0, 3.0]$ | $[4.0, 6.0]$ | OOD strictly $> 3.0\text{ N}$ |
| **Impulse Duration** | $\Delta t_{\text{imp}}$ | $\text{s}$ | $0.10$ ($1$ RL step) | $0.10$ ($1$ RL step) | $0.10$ ($1$ RL step) | Fixed discrete pulse |
| **Impulse Onset** | $t_{\text{imp}}$ | $\text{s}$ | $[1.5, 3.5]$ | $[1.0, 4.0]$ | $[1.0, 3.5]$ | Randomized |
| **Sensor Noise Scale** | $k_{\text{noise}}$ | $\times$ | $[1.0, 2.0]$ | $[1.0, 2.0]$ | $[3.5, 5.0]$ | OOD strictly $> 2.0\times$ |
| **Motor Degradation (LoE)** | $\delta_{\text{motor}}$ | fraction | $[0.0, 0.25]$ | $[0.0, 0.25]$ | $[0.30, 0.70]$ | **Strict Separation:** Train $\le 25\%$, OOD $\ge 30\%$ |
| **Degraded Motors** | $\mathcal{I}_{\text{deg}}$ | indices | $\{0\}, \{1\}, \{2\}, \{3\}$ (1-2 rotors) | Random 2 rotors | $\{0, 1\}$ or $\{1, 2\}$ (asymmetric pair) | Asymmetric cross-coupling |
| **Mass Scale** | $m_{\text{scale}}$ | $\times$ | $[0.90, 1.15]$ | $[0.90, 1.15]$ | $[1.20, 1.50]$ | OOD strictly $> 1.15\times$ |
| **Inertia Scale** | $I_{\text{scale}}$ | $\times$ | $[0.90, 1.15]$ | $[0.90, 1.15]$ | $[1.20, 1.50]$ | OOD strictly $> 1.15\times$ |

---

## 4. Split Specifications and Allocation

```
                              ┌──────────────────────────────────┐
                              │    MASTER TASK DATASET: N=130    │
                              │       Master Seed: S_0 = 42      │
                              └─────────────────┬────────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
  ┌──────────────────────────────┐┌──────────────────────────────┐┌──────────────────────────────┐
  │  META-TRAIN (N_train = 80)   ││   META-VAL (N_val = 20)      ││   OOD-TEST (N_ood = 30)      │
  │  Seeds: [1000, 1079]         ││   Seeds: [2000, 2019]        ││   Seeds: [3000, 3029]        │
  │  Waypoint Seeds: [10000..]   ││   Waypoint Seeds: [20000..]  ││   Waypoint Seeds: [30000..]  │
  │  - 16 Light Wind             ││   - In-distribution          ││   - 6 Extreme Wind/Gusts     │
  │  - 16 Gust/Turbulence        ││     interpolation            ││   - 6 High Impulse Shocks    │
  │  - 16 Impulse Perturbations  ││   - Novel waypoint geometries││   - 6 Extreme Sensor Noise   │
  │  - 16 Sensor Noise Attacks   ││   - Mixed compound tasks     ││   - 6 Severe Rotor LoE       │
  │  - 16 Mild Actuator LoE      ││                              ││   - 6 Multi-Modal Compound   │
  └──────────────────────────────┘└──────────────────────────────┘└──────────────────────────────┘
```

---

## 5. Prevention of Train / Validation / Test Data Leakage

To prevent methodological data leakage:
1. **Disjoint Pseudo-Random Stream Offsets:** Training, validation, and test tasks use dedicated, non-overlapping seed ranges ($1000+i$, $2000+i$, $3000+i$).
2. **Disjoint Waypoint Geometries:** Obstacle placements and target positions are seeded independently across splits ($10000+i$, $20000+i$, $30000+i$).
3. **Strict Range Disjointness:** Out-of-distribution physical parameters (e.g. wind $> 3.0\text{ m/s}$, motor degradation $\ge 30\%$) never appear in training tasks.
4. **Frozen Test Set Rule:** The 30 OOD test tasks are frozen in `configs/meta_tasks.yaml` and never accessed during meta-training or hyperparameter tuning.

---

## 6. Compound Disturbance Formulation

Realistic flight involves compound, concurrent perturbations. The generator explicitly marks multi-modal tasks via `compound_disturbance_flags`:
* **Wind + Sensor Corruption:** High aerodynamic drag coupled with noisy attitude estimation.
* **Wind + Actuator Degradation:** Asymmetric motor thrust loss in the presence of continuous crosswind.
* **Impulse + Actuator Degradation:** Sudden lateral force pulse applied to an already degraded quadrotor.
* **Full Multi-Modal Compound:** Severe wind ($4.0\text{ m/s}$) + $4.0\times$ sensor noise + $40\%$ motor degradation + random impulse shock.

---

## 7. Controlled Motor Degradation Sweep Specification

For the controlled actuator loss sweep in Phase 20, the generator provides dedicated evaluation instances across $\delta \in \{0\%, 10\%, 20\%, 30\%, 40\%, 50\%, 60\%, 70\%\}$:
* **Nominal:** $\delta = 0\%$ (Full control authority).
* **Moderate (In-Distribution):** $\delta \in \{10\%, 20\%\}$.
* **Boundary Condition:** $\delta = 25\%$ (Training upper bound).
* **Severe (OOD Test):** $\delta \in \{30\%, 40\%, 50\%, 60\%, 70\%\}$.

---

## 8. Validation Rules and Assertion Pipeline

The validation pipeline `validate_task_manifest()` enforces:
1. Exact count verification ($|\text{Train}| = 80$, $|\text{Val}| = 20$, $|\text{OOD}| = 30$, $|\text{Total}| = 130$).
2. Unique Task ID verification across the global manifest.
3. Strict seed namespace disjointness: $\text{Seeds}_{\text{Train}} \cap \text{Seeds}_{\text{Val}} \cap \text{Seeds}_{\text{OOD}} = \emptyset$.
4. Boundary checks on all continuous physical variables.
5. Strict OOD separation assertion ($\delta_{\text{motor}} \ge 0.30$ on all OOD degraded tasks).

---

## 9. Serialization and Downstream Integration

All generated tasks inherit standard dictionary and JSON serialization via `MetaTask.to_dict()` and `MetaTask.to_json()`. The manifest is exported to `configs/meta_tasks_manifest.json` for deterministic execution by evaluation and training scripts.
