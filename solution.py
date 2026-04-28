"""
Boom Challenge — Trajectory Unknown
Author: Goutam Roy
         Freelancer: https://www.freelancer.com/u/Goutam895
         Submission Type: Individual (Solo)
Model: XGBoost MultiOutput Regressor

Fixes applied:
  [1] fines_frac / oversize_frac clipped to 0 (no negative fractions)
  [2] Proper train/validation split for honest R2 reporting
  [3] Inverse design: batch-predict filter + energy-sort
  [4] All inverse design inputs clipped to declared bounds
  [5] Post-clip model re-verification pass
  [6] Final strict re-verify — only truly valid rows kept
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json
import warnings
warnings.filterwarnings('ignore')

BASE = 'Boom-Challenge-Datasets-main'

FEATURES = ['energy','angle_rad','coupling','strength',
            'porosity','gravity','atmosphere','shape_factor']
TARGETS  = ['P80','fines_frac','oversize_frac',
            'R95','R50_fines','R50_oversize']

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("Loading data...")
train  = pd.read_csv(f'{BASE}/forward_prediction/train.csv')
labels = pd.read_csv(f'{BASE}/forward_prediction/train_labels.csv')
test   = pd.read_csv(f'{BASE}/forward_prediction/test.csv')

X = train[FEATURES].values
y = labels[TARGETS].values
X_test = test[FEATURES].values

with open(f'{BASE}/inverse_design/constraints.json') as f:
    cfg = json.load(f)
bounds_cfg = cfg['input_bounds']

# ── 2. Train/Val Split [Fix #2] ────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ── 3. Train XGBoost ───────────────────────────────────────────────────────────
print("\nTraining model...")
model = MultiOutputRegressor(XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0))
model.fit(X_tr, y_tr)

r2_val = r2_score(y_val, model.predict(X_val), multioutput='raw_values')
print("  Validation R2 (unseen 20% data):")
for t, s in zip(TARGETS, r2_val):
    print(f"    {t}: {s:.4f}")

# Retrain on full data for best predictions
model.fit(X, y)

# ── 4. Forward Prediction ─────────────────────────────────────────────────────
print("\nGenerating forward predictions...")
sub = pd.DataFrame(model.predict(X_test), columns=TARGETS)
sub['fines_frac']    = sub['fines_frac'].clip(lower=0)     # Fix #1
sub['oversize_frac'] = sub['oversize_frac'].clip(lower=0)  # Fix #1
sub.insert(0, 'scenario_id', range(len(sub)))
sub.to_csv('prediction_submission.csv', index=False)
print(f"  Saved prediction_submission.csv ({len(sub)} rows) — no negatives ✓")

# ── 5. Inverse Design ─────────────────────────────────────────────────────────
print("\nInverse design...")

# Step A: batch predict training rows, filter by output constraints
preds_all = model.predict(train[FEATURES].values)
mask = (preds_all[:,0] >= 96) & (preds_all[:,0] <= 101) & (preds_all[:,3] <= 175)
valid_inputs = train[FEATURES][mask].copy()

# Step B: clip ALL features to declared input bounds [Fix #4]
for col in FEATURES:
    valid_inputs[col] = valid_inputs[col].clip(
        lower=bounds_cfg[col]['min'],
        upper=bounds_cfg[col]['max']
    )

# Step C: sort by energy ascending (lower energy = better small-impact score)
valid_inputs = valid_inputs.sort_values('energy').reset_index(drop=True)

# Step D: re-verify after clipping [Fix #5]
preds_check = model.predict(valid_inputs[FEATURES].values)
valid_mask = (
    (preds_check[:,0] >= 96) & (preds_check[:,0] <= 101) & (preds_check[:,3] <= 175)
)
design_df = valid_inputs[valid_mask].head(20).reset_index(drop=True)

# Step E: final strict re-verify — remove any remaining edge cases [Fix #6]
fp = model.predict(design_df[FEATURES].values)
truly_valid = (fp[:,0] >= 96) & (fp[:,0] <= 101) & (fp[:,3] <= 175)
design_df = design_df[truly_valid].reset_index(drop=True)
design_df.insert(0, 'submission_id', range(len(design_df)))
design_df.to_csv('design_submission.csv', index=False)

# ── 6. Final Report ────────────────────────────────────────────────────────────
fp2 = model.predict(design_df[FEATURES].values)
vc  = sum(1 for p,r in zip(fp2[:,0],fp2[:,3]) if 96<=p<=101 and r<=175)
bounds_ok = all(
    (design_df[col] >= bounds_cfg[col]['min']).all() and
    (design_df[col] <= bounds_cfg[col]['max']).all()
    for col in FEATURES
)

# Save trained model
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("model.pkl saved!")

print(f"  P80:  [{fp2[:,0].min():.4f}, {fp2[:,0].max():.4f}]  (target: 96–101) ✓")
print(f"  R95 max:  {fp2[:,3].max():.4f}  (target: ≤175) ✓")
print(f"  Valid: {vc}/{len(design_df)} ✓")
print(f"  All input bounds satisfied: {bounds_ok} ✓")
print(f"  Saved design_submission.csv ({len(design_df)} scenarios)")
print("\nAll done!")
