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

print("Testing hyperparameter tuning...\n")

best_mae = float('inf')
best_config = {}
best_model = None

# XGBoost tuning
print("Tuning XGBoost...")
xgb_configs = [
    {"lr": 0.05, "depth": 5, "est": 800},
    {"lr": 0.05, "depth": 6, "est": 800},
    {"lr": 0.1, "depth": 5, "est": 600},
    {"lr": 0.1, "depth": 6, "est": 600},
]

for config in xgb_configs:
    xgb = XGBRegressor(
        n_estimators=config["est"],
        learning_rate=config["lr"],
        max_depth=config["depth"],
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:absoluteerror",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    print(f"  lr={config['lr']}, depth={config['depth']}, est={config['est']}: MAE={mae:.4f}")
    
    if mae < best_mae:
        best_mae = mae
        best_config = config
        best_model = xgb

print(f"\n Best XGBoost config: {best_config} with MAE: {best_mae:.4f}\n")

# LightGBM tuning
print("Tuning LightGBM...")
lgb_configs = [
    {"lr": 0.05, "depth": 6, "est": 800},
    {"lr": 0.05, "depth": 7, "est": 800},
    {"lr": 0.1, "depth": 6, "est": 600},
    {"lr": 0.1, "depth": 7, "est": 600},
]

for config in lgb_configs:
    lgb = LGBMRegressor(
        n_estimators=config["est"],
        learning_rate=config["lr"],
        max_depth=config["depth"],
        subsample=0.9,
        colsample_bytree=0.9,
        objective="mae",
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(X_train, y_train)
    pred = lgb.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    print(f"  lr={config['lr']}, depth={config['depth']}, est={config['est']}: MAE={mae:.4f}")
    
    if mae < best_mae:
        best_mae = mae
        best_config = config
        best_model = lgb

print(f"\nBest LightGBM config: {best_config} with MAE: {best_mae:.4f}\n")

print("="*50)
print(f"BEST MAE: {best_mae:.4f}")
print(f"CONFIG: {best_config}")
print("="*50)

# Predictions on test set
test_pred = best_model.predict(preprocess.transform(X_test))

# Save submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Age': test_pred
})
submission.to_csv('submission_tuned.csv', index=False)
print(f"\nSubmission saved: submission_tuned.csv")
