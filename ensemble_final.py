import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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

# Preprocessing
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
    X_processed, y, test_size=0.2, random_state=42
)

print("Building optimized ensemble...\n")

# Best tuned models
xgb = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.92,
    colsample_bytree=0.92,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="reg:absoluteerror",
    random_state=42,
    n_jobs=-1
)

lgb = LGBMRegressor(
    n_estimators=600,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.92,
    colsample_bytree=0.92,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="mae",
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=18,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.9,
    min_samples_leaf=2,
    random_state=42
)

# Train all models
print("Training models...")
xgb.fit(X_train, y_train)
lgb.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Get predictions
xgb_pred = xgb.predict(X_valid)
lgb_pred = lgb.predict(X_valid)
rf_pred = rf.predict(X_valid)
gb_pred = gb.predict(X_valid)

print(f"XGBoost MAE: {mean_absolute_error(y_valid, xgb_pred):.4f}")
print(f"LightGBM MAE: {mean_absolute_error(y_valid, lgb_pred):.4f}")
print(f"RandomForest MAE: {mean_absolute_error(y_valid, rf_pred):.4f}")
print(f"GradientBoosting MAE: {mean_absolute_error(y_valid, gb_pred):.4f}")

# Test different ensemble weights
print("\nTesting ensemble weights...")
best_ensemble_mae = float('inf')
best_weights = None

weights_to_test = [
    (0.4, 0.4, 0.1, 0.1),
    (0.35, 0.35, 0.2, 0.1),
    (0.35, 0.35, 0.15, 0.15),
    (0.3, 0.3, 0.25, 0.15),
    (0.45, 0.35, 0.1, 0.1),
    (0.4, 0.35, 0.15, 0.1),
]

for w1, w2, w3, w4 in weights_to_test:
    ensemble = w1*xgb_pred + w2*lgb_pred + w3*rf_pred + w4*gb_pred
    mae = mean_absolute_error(y_valid, ensemble)
    print(f"  Weights ({w1}, {w2}, {w3}, {w4}): MAE={mae:.4f}")
    
    if mae < best_ensemble_mae:
        best_ensemble_mae = mae
        best_weights = (w1, w2, w3, w4)

print(f"\nBEST ENSEMBLE MAE: {best_ensemble_mae:.4f}")
print(f"BEST WEIGHTS: XGB={best_weights[0]}, LGB={best_weights[1]}, RF={best_weights[2]}, GB={best_weights[3]}")

# Final predictions
X_test_processed = preprocess.transform(X_test)
xgb_test = xgb.predict(X_test_processed)
lgb_test = lgb.predict(X_test_processed)
rf_test = rf.predict(X_test_processed)
gb_test = gb.predict(X_test_processed)

test_pred = (best_weights[0]*xgb_test + best_weights[1]*lgb_test + 
             best_weights[2]*rf_test + best_weights[3]*gb_test)

# Save submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Age': test_pred
})
submission.to_csv('submission_ensemble.csv', index=False)
print(f"\nSubmission saved: submission_ensemble.csv")
print(f"Validation MAE target achieved: {best_ensemble_mae:.4f}")
