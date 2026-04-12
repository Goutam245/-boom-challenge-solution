# Boom Challenge — Trajectory Unknown
### Physics-informed AI/ML solution for asteroid ejecta prediction

**Submitted by:** Goutam Roy                       
**Submission Type:** Individual (Solo)  
**Challenge:** [Boom Challenge on Freelancer.com](https://www.freelancer.com/boom)

---

## Overview

This is a solo submission by **Goutam Roy** for the **Boom Challenge**, covering both parts of the challenge:

- **Part 1 — Forward Prediction:** Predict 6 ejecta outcomes from 8 impact parameters
- **Part 2 — Inverse Design:** Propose 20 impact scenarios satisfying given output constraints

All code, modeling, and submission files were developed independently by a single individual.

---

## Algorithm

### Model: XGBoost Multi-Output Regressor

XGBoost was chosen for its ability to capture non-linear interactions between physics parameters (e.g., energy × coupling, gravity × angle). A `MultiOutputRegressor` trains one gradient-boosted tree per output target.

**Validation R² Scores (80/20 train-val split, unseen data):**

| Target | Validation R² |
|---|---|
| P80 | 0.9730 |
| fines_frac | 0.9416 |
| oversize_frac | 0.9884 |
| R95 | 0.9097 |
| R50_fines | 0.8820 |
| R50_oversize | 0.8417 |

All scores measured on a held-out 20% validation set not seen during training.

### Key Implementation Details

- `fines_frac` and `oversize_frac` are clipped to `≥ 0` (physical fractions cannot be negative)
- Model is evaluated on a validation split first, then retrained on full data before generating test predictions
- Inverse design uses batch prediction over training rows to identify constraint-satisfying input regions
- All inverse design inputs are clipped to declared bounds from `constraints.json`
- A post-clip model re-verification pass (Steps D & E) ensures only truly valid scenarios are kept

---

## Results

### Forward Prediction
- 492 test scenarios predicted
- Output: `prediction_submission.csv`
- No negative fraction values ✓

### Inverse Design
- 20 valid scenarios, all satisfying:
  - P80 ∈ [96.21, 100.63] ✅ (target: 96–101)
  - R95 ≤ 111.46 ✅ (target: ≤175)
  - All input features within declared bounds ✅
- Output: `design_submission.csv`

---

## Repository Structure

```
├── solution.py                        # Complete solution script
├── prediction_submission.csv          # Forward prediction (492 rows)
├── design_submission.csv              # Inverse design (20 scenarios)
├── README.md
├── VIDEO_SCRIPT.txt
└── Boom-Challenge-Datasets-main/
    ├── forward_prediction/
    │   ├── train.csv
    │   ├── train_labels.csv
    │   └── test.csv
    └── inverse_design/
        └── constraints.json
```

---

## How to Run

```bash
pip install xgboost scikit-learn pandas numpy
python solution.py
```

Outputs: `prediction_submission.csv` and `design_submission.csv`

---

## Physics Rationale

- **Energy + Coupling** → drive fragment size (higher = larger P80)
- **Gravity** → constrains ejecta range (higher gravity = smaller R95)
- **Porosity + Strength** → control fragmentation into fines
- **Angle** → affects ejecta directionality and spread
- **Shape factor** → higher = more irregular fragments (larger oversize_frac)

For inverse design, low-energy + high-gravity parameter regions produce P80 ~98mm with R95 well under 175m — satisfying both output constraints while minimizing impact energy for the best small-impact score.

---

## Contact

**Goutam Roy** 
