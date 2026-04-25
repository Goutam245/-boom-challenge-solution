# Boom Challenge — Trajectory Unknown
### Physics-informed AI/ML solution for asteroid ejecta prediction

**Submitted by:** Goutam Roy  
**Freelancer Profile:** [@Goutam895](https://www.freelancer.com/u/Goutam895)  
**Submission Type:** Individual (Solo)  
**Challenge:** [Boom Challenge on Freelancer.com](https://www.freelancer.com/boom)

---

## Overview

This is a solo submission by **Goutam Roy** for the **Boom Challenge**, covering both parts:

- **Part 1 — Forward Prediction:** Predict 6 ejecta outcomes from 8 impact parameters
- **Part 2 — Inverse Design:** Propose 20 impact scenarios satisfying given output constraints

All code, modeling, and submission files were developed independently.

---

## How to Reproduce

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Place dataset
Make sure `Boom-Challenge-Datasets-main/` folder is in the same directory as `solution.py`.

### Step 3 — Run
```bash
python solution.py
```

### Output files generated
| File | Description |
|---|---|
| `prediction_submission.csv` | Forward prediction — 492 test scenarios |
| `design_submission.csv` | Inverse design — 20 scenarios |

---

## Software Requirements

See `requirements.txt` for exact versions.

| Package | Version |
|---|---|
| Python | 3.10+ |
| xgboost | 3.2.0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.1 |
| numpy | 2.4.3 |
| scipy | 1.17.1 |

No GPU required. No special drivers or CUDA needed.

---

## Hardware Requirements

| Spec | Detail |
|---|---|
| CPU | Any modern CPU (Intel/AMD) |
| RAM | 2 GB minimum |
| GPU | Not required |
| Training time | ~25 seconds on standard laptop CPU |
| OS | Windows / Linux / macOS |
| Cloud cost | $0 — runs locally |

---

## Algorithm

### Model: XGBoost Multi-Output Regressor

XGBoost was chosen for its ability to capture non-linear physical interactions between impact parameters (e.g., energy x coupling, gravity x angle). A `MultiOutputRegressor` wrapper trains one independent gradient-boosted tree per output target.

### Physics Rationale

The model predicts ejecta outcomes from the target surface only. The projectile is not modeled as a separate fragmenting object.

| Parameter | Physical Role |
|---|---|
| Energy + Coupling | Drive fragment size — higher energy with efficient coupling produces larger P80 |
| Gravity | Constrains ejecta range — higher gravity pulls fragments back, reducing R95 |
| Porosity + Strength | Control fragmentation into fines — porous, weak material produces more fines_frac |
| Angle | Affects directionality — grazing impacts produce more asymmetric spread |
| Shape factor | Controls irregularity — higher value produces more oversize fragments |

### Validation R2 Scores (80/20 train-val split, unseen data)

| Target | Validation R2 |
|---|---|
| P80 | 0.9730 |
| fines_frac | 0.9416 |
| oversize_frac | 0.9884 |
| R95 | 0.9097 |
| R50_fines | 0.8820 |
| R50_oversize | 0.8417 |

All scores are from the held-out 20% validation set — not seen during training.

### Key Implementation Details

- `fines_frac` and `oversize_frac` clipped to >= 0 (physical fractions cannot be negative)
- Model evaluated on validation split first, then retrained on full dataset before test predictions
- Inverse design uses batch prediction over training rows to identify constraint-satisfying regions
- All inverse design inputs clipped to declared bounds from `constraints.json`
- Two-pass re-verification ensures only truly valid scenarios are kept

### Assumptions and Limitations

- Model assumes Mox-95 physics is self-consistent with training data distribution
- XGBoost does not extrapolate well beyond training feature range — out-of-distribution test scenarios may have higher error
- Stochastic nature of real impacts is not modeled — predictions represent average outcomes
- Inverse design relies on training data proximity — parameter regions with no training coverage may be missed

### Extensibility

The same pipeline applies to any tabular physics simulation dataset with defined inputs and outputs. Candidate applications: volcanic ejecta modeling, building collapse debris prediction, landslide runout estimation, underwater explosion fragment dispersal. Only the dataset and constraint file need to change.

---

## Results

### Forward Prediction
- 492 test scenarios predicted
- Output: `prediction_submission.csv`
- No negative fraction values

### Inverse Design
- 20 valid scenarios, all satisfying:
  - P80 between 96.21 and 100.63 (target: 96–101)
  - R95 max 111.46 (target: <= 175)
  - All input features within declared bounds
- Output: `design_submission.csv`

---

## Repository Structure
boom-challenge-solution/
├── solution.py
├── requirements.txt
├── prediction_submission.csv
├── design_submission.csv
├── README.md
└── Boom-Challenge-Datasets-main/
├── forward_prediction/
│   ├── train.csv
│   ├── train_labels.csv
│   └── test.csv
└── inverse_design/
└── constraints.json

---

## Contact

**Goutam Roy**  
Freelancer: [@Goutam895](https://www.freelancer.com/u/Goutam895)
