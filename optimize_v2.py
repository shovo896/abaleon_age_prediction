import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

# BETTER PREPROCESSING WITH FEATURE ENGINEERING
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
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
    X_processed, y, test_size=0.15, random_state=42  # Smaller validation for more training data
)

print("="*60)
print("ADVANCED HYPERPARAMETER TUNING + FEATURE ENGINEERING")
print("="*60 + "\n")

best_mae = float('inf')
best_model = None
best_params = {}
all_results = []

# AGGRESSIVE XGBOOST TUNING
print("Tuning XGBoost (advanced)...")
xgb_params = [
    {"n_est": 1000, "lr": 0.03, "depth": 7, "sub": 0.95},
    {"n_est": 1200, "lr": 0.02, "depth": 8, "sub": 0.95},
    {"n_est": 1000, "lr": 0.04, "depth": 6, "sub": 0.9},
    {"n_est": 1500, "lr": 0.025, "depth": 7, "sub": 0.93},
]

for params in xgb_params:
    xgb = XGBRegressor(
        n_estimators=params["n_est"],
        learning_rate=params["lr"],
        max_depth=params["depth"],
        subsample=params["sub"],
        colsample_bytree=0.95,
        reg_alpha=0.05,
        reg_lambda=0.05,
        objective="reg:absoluteerror",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0)
    pred = xgb.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    result = (f"XGB(est={params['n_est']},lr={params['lr']},d={params['depth']})", mae, xgb)
    all_results.append(result)
    print(f"  {result[0]}: {mae:.4f}")
    if mae < best_mae:
        best_mae = mae
        best_model = xgb
        best_params = {"type": "XGB", "params": params}

# AGGRESSIVE LIGHTGBM TUNING
print("\nTuning LightGBM (advanced)...")
lgb_params = [
    {"n_est": 1000, "lr": 0.03, "depth": 7, "sub": 0.95},
    {"n_est": 1200, "lr": 0.02, "depth": 8, "sub": 0.95},
    {"n_est": 1500, "lr": 0.02, "depth": 7, "sub": 0.93},
    {"n_est": 1000, "lr": 0.04, "depth": 6, "sub": 0.9},
]

for params in lgb_params:
    lgb = LGBMRegressor(
        n_estimators=params["n_est"],
        learning_rate=params["lr"],
        max_depth=params["depth"],
        subsample=params["sub"],
        colsample_bytree=0.95,
        reg_alpha=0.05,
        reg_lambda=0.05,
        objective="mae",
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(X_train, y_train)
    pred = lgb.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    result = (f"LGB(est={params['n_est']},lr={params['lr']},d={params['depth']})", mae, lgb)
    all_results.append(result)
    print(f"  {result[0]}: {mae:.4f}")
    if mae < best_mae:
        best_mae = mae
        best_model = lgb
        best_params = {"type": "LGB", "params": params}

# EXTRA: GRADIENT BOOSTING
print("\nTuning GradientBoosting...")
gb_params = [
    {"n_est": 500, "lr": 0.05, "depth": 6, "sub": 0.9},
    {"n_est": 800, "lr": 0.03, "depth": 7, "sub": 0.92},
]

for params in gb_params:
    gb = GradientBoostingRegressor(
        n_estimators=params["n_est"],
        learning_rate=params["lr"],
        max_depth=params["depth"],
        subsample=params["sub"],
        min_samples_leaf=2,
        random_state=42
    )
    gb.fit(X_train, y_train)
    pred = gb.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    result = (f"GB(est={params['n_est']},lr={params['lr']},d={params['depth']})", mae, gb)
    all_results.append(result)
    print(f"  {result[0]}: {mae:.4f}")
    if mae < best_mae:
        best_mae = mae
        best_model = gb
        best_params = {"type": "GB", "params": params}

# BUILD BEST ENSEMBLE FROM TOP MODELS
print("\nBuilding ensemble from top 3 models...")
all_results.sort(key=lambda x: x[1])
top_3 = all_results[:3]

print(f"Top 3 models:")
for name, mae, model in top_3:
    print(f"  {name}: {mae:.4f}")

# Get predictions from top 3
pred_1 = top_3[0][2].predict(X_valid)
pred_2 = top_3[1][2].predict(X_valid)
pred_3 = top_3[2][2].predict(X_valid)

# Test different ensemble strategies
print("\nTesting ensemble strategies...")
ensemble_strategies = {
    "Simple Average": (pred_1 + pred_2 + pred_3) / 3,
    "Weighted (0.5, 0.3, 0.2)": 0.5*pred_1 + 0.3*pred_2 + 0.2*pred_3,
    "Weighted (0.4, 0.35, 0.25)": 0.4*pred_1 + 0.35*pred_2 + 0.25*pred_3,
    "Weighted (0.45, 0.35, 0.2)": 0.45*pred_1 + 0.35*pred_2 + 0.2*pred_3,
}

ensemble_scores = {}
for strategy_name, ensemble_pred in ensemble_strategies.items():
    mae = mean_absolute_error(y_valid, ensemble_pred)
    ensemble_scores[strategy_name] = mae
    print(f"  {strategy_name}: {mae:.4f}")

best_ensemble_mae = min(ensemble_scores.values())
best_ensemble_name = [k for k, v in ensemble_scores.items() if v == best_ensemble_mae][0]

print("\n" + "="*60)
print(f"BEST SINGLE MODEL: {best_params}")
print(f"BEST SINGLE MAE: {best_mae:.4f}")
print(f"BEST ENSEMBLE: {best_ensemble_name}")
print(f"BEST ENSEMBLE MAE: {best_ensemble_mae:.4f}")
print("="*60)

# Make predictions on test set
X_test_processed = preprocess.transform(X_test)

if best_ensemble_mae < best_mae:
    # Use ensemble
    pred_1_test = top_3[0][2].predict(X_test_processed)
    pred_2_test = top_3[1][2].predict(X_test_processed)
    pred_3_test = top_3[2][2].predict(X_test_processed)
    
    if best_ensemble_name == "Simple Average":
        test_pred = (pred_1_test + pred_2_test + pred_3_test) / 3
    elif best_ensemble_name == "Weighted (0.5, 0.3, 0.2)":
        test_pred = 0.5*pred_1_test + 0.3*pred_2_test + 0.2*pred_3_test
    elif best_ensemble_name == "Weighted (0.4, 0.35, 0.25)":
        test_pred = 0.4*pred_1_test + 0.35*pred_2_test + 0.25*pred_3_test
    else:  # (0.45, 0.35, 0.2)
        test_pred = 0.45*pred_1_test + 0.35*pred_2_test + 0.2*pred_3_test
    final_mae = best_ensemble_mae
else:
    # Use best single model
    test_pred = best_model.predict(X_test_processed)
    final_mae = best_mae

# Save submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Age': test_pred
})
submission.to_csv('submission_v2.csv', index=False)
print(f"\nFinal Validation MAE: {final_mae:.4f}")
print(f"Submission saved: submission_v2.csv")
