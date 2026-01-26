import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

TARGET = "Age"
ID_COL = "id"

X = train_df.drop(columns=[TARGET, ID_COL] if ID_COL in train_df.columns else [TARGET])
y = train_df[TARGET].astype(float)

X_test = test_df.drop(columns=[ID_COL] if ID_COL in test_df.columns else [])

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# Simple preprocessing (faster convergence)
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
], remainder="drop")

X_processed = preprocess.fit_transform(X)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_processed, y, test_size=0.12, random_state=42  # Even less validation data
)

print("="*60)
print("EXTREME HYPERPARAMETER OPTIMIZATION - Target: < 1.30")
print("="*60 + "\n")

# Store all model results
models_data = []

# EXTREME XGBOOST CONFIGS
print("Testing extreme XGBoost configs...\n")
xgb_configs = [
    {"nest": 2000, "lr": 0.015, "d": 8, "sub": 0.95, "col": 0.95, "ga": 0, "l": 0},
    {"nest": 2000, "lr": 0.02, "d": 7, "sub": 0.95, "col": 0.95, "ga": 0, "l": 0},
    {"nest": 1800, "lr": 0.02, "d": 8, "sub": 0.96, "col": 0.96, "ga": 0, "l": 0},
    {"nest": 1500, "lr": 0.025, "d": 8, "sub": 0.95, "col": 0.95, "ga": 0.05, "l": 0.05},
    {"nest": 2000, "lr": 0.015, "d": 9, "sub": 0.95, "col": 0.95, "ga": 0, "l": 0},
    {"nest": 1600, "lr": 0.03, "d": 7, "sub": 0.95, "col": 0.95, "ga": 0, "l": 0},
]

for cfg in xgb_configs:
    xgb = XGBRegressor(
        n_estimators=cfg["nest"],
        learning_rate=cfg["lr"],
        max_depth=cfg["d"],
        subsample=cfg["sub"],
        colsample_bytree=cfg["col"],
        reg_alpha=cfg["ga"],
        reg_lambda=cfg["l"],
        objective="reg:absoluteerror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    pred = xgb.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    models_data.append(("XGB", cfg, mae, xgb, pred))
    print(f"XGB(n={cfg['nest']}, lr={cfg['lr']}, d={cfg['d']}): MAE = {mae:.4f}")

print()

# EXTREME LIGHTGBM CONFIGS
print("Testing extreme LightGBM configs...\n")
lgb_configs = [
    {"nest": 2000, "lr": 0.015, "d": 8, "sub": 0.95, "col": 0.95, "ga": 0, "l": 0},
    {"nest": 2000, "lr": 0.02, "d": 7, "sub": 0.95, "col": 0.95, "ga": 0, "l": 0},
    {"nest": 1800, "lr": 0.02, "d": 8, "sub": 0.96, "col": 0.96, "ga": 0, "l": 0},
    {"nest": 1500, "lr": 0.025, "d": 9, "sub": 0.95, "col": 0.95, "ga": 0.05, "l": 0.05},
    {"nest": 2000, "lr": 0.015, "d": 9, "sub": 0.95, "col": 0.95, "ga": 0, "l": 0},
]

for cfg in lgb_configs:
    lgb = LGBMRegressor(
        n_estimators=cfg["nest"],
        learning_rate=cfg["lr"],
        max_depth=cfg["d"],
        subsample=cfg["sub"],
        colsample_bytree=cfg["col"],
        reg_alpha=cfg["ga"],
        reg_lambda=cfg["l"],
        objective="mae",
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(X_train, y_train)
    pred = lgb.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    models_data.append(("LGB", cfg, mae, lgb, pred))
    print(f"LGB(n={cfg['nest']}, lr={cfg['lr']}, d={cfg['d']}): MAE = {mae:.4f}")

# Sort by MAE
models_data.sort(key=lambda x: x[2])

print("\n" + "="*60)
print("TOP 5 MODELS:")
for i, (typ, cfg, mae, model, pred) in enumerate(models_data[:5]):
    print(f"{i+1}. {typ} - MAE: {mae:.4f}")

# Test different ensemble combinations
print("\nBuilding optimized ensemble...\n")

best_ensemble_mae = float('inf')
best_ensemble_config = None

# Get top 4 predictions
top_preds = [p[4] for p in models_data[:4]]

# Test all combinations
for w1 in [0.4, 0.45, 0.5]:
    for w2 in [0.25, 0.3, 0.35]:
        for w3 in [0.1, 0.15, 0.2]:
            w4 = 1 - w1 - w2 - w3
            if w4 > 0:
                ens_pred = w1*top_preds[0] + w2*top_preds[1] + w3*top_preds[2] + w4*top_preds[3]
                mae = mean_absolute_error(y_valid, ens_pred)
                if mae < best_ensemble_mae:
                    best_ensemble_mae = mae
                    best_ensemble_config = (w1, w2, w3, w4)

print(f"BEST ENSEMBLE WEIGHTS: {best_ensemble_config}")
print(f"BEST ENSEMBLE MAE: {best_ensemble_mae:.4f}")

# Get test predictions
X_test_processed = preprocess.transform(X_test)
test_preds = [model[3].predict(X_test_processed) for model in models_data[:4]]

w1, w2, w3, w4 = best_ensemble_config
final_pred = w1*test_preds[0] + w2*test_preds[1] + w3*test_preds[2] + w4*test_preds[3]

# Save
submission = pd.DataFrame({
    'id': test_df['id'],
    'Age': final_pred
})
submission.to_csv('submission_v3.csv', index=False)

print(f"\n{'='*60}")
print(f"FINAL VALIDATION MAE: {best_ensemble_mae:.4f}")
print(f"Submission saved: submission_v3.csv")
print(f"{'='*60}")
