import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
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

print("Testing different models...\n")

# Model 1: XGBoost (fast, good)
print("1. XGBoost...")
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:absoluteerror",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_valid)
xgb_mae = mean_absolute_error(y_valid, xgb_pred)
print(f"   XGBoost MAE: {xgb_mae:.4f}\n")

# Model 2: LightGBM (faster)
print("2. LightGBM...")
lgb = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="mae",
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict(X_valid)
lgb_mae = mean_absolute_error(y_valid, lgb_pred)
print(f"   LightGBM MAE: {lgb_mae:.4f}\n")

# Model 3: RandomForest (good baseline)
print("3. RandomForest...")
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_valid)
rf_mae = mean_absolute_error(y_valid, rf_pred)
print(f"   RandomForest MAE: {rf_mae:.4f}\n")

# Model 4: Gradient Boosting
print("4. GradientBoosting...")
gb = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.9,
    random_state=42
)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_valid)
gb_mae = mean_absolute_error(y_valid, gb_pred)
print(f"   GradientBoosting MAE: {gb_mae:.4f}\n")

# Weighted ensemble
print("5. Weighted Ensemble...")
ensemble_pred = (xgb_pred * 0.3 + lgb_pred * 0.3 + rf_pred * 0.2 + gb_pred * 0.2)
ensemble_mae = mean_absolute_error(y_valid, ensemble_pred)
print(f"   Ensemble MAE: {ensemble_mae:.4f}\n")

# Find best
scores = {
    "XGBoost": xgb_mae,
    "LightGBM": lgb_mae,
    "RandomForest": rf_mae,
    "GradientBoosting": gb_mae,
    "Ensemble": ensemble_mae
}

best_model = min(scores, key=scores.get)
print("="*50)
print(f"BEST MODEL: {best_model} with MAE: {scores[best_model]:.4f}")
print("="*50)

# Predictions on test set
if best_model == "XGBoost":
    test_pred = xgb.predict(preprocess.transform(X_test))
elif best_model == "LightGBM":
    test_pred = lgb.predict(preprocess.transform(X_test))
elif best_model == "RandomForest":
    test_pred = rf.predict(preprocess.transform(X_test))
elif best_model == "GradientBoosting":
    test_pred = gb.predict(preprocess.transform(X_test))
else:  # Ensemble
    xgb_test = xgb.predict(preprocess.transform(X_test))
    lgb_test = lgb.predict(preprocess.transform(X_test))
    rf_test = rf.predict(preprocess.transform(X_test))
    gb_test = gb.predict(preprocess.transform(X_test))
    test_pred = (xgb_test * 0.3 + lgb_test * 0.3 + rf_test * 0.2 + gb_test * 0.2)

# Save submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Age': test_pred
})
submission.to_csv('submission_optimized.csv', index=False)
print(f"\nSubmission saved: submission_optimized.csv")
