import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor

# -----------------------
# 1) Load data
# -----------------------
train_path = "train.csv"
test_path  = "test.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

# target
y = train["Age"].astype(float)
X = train.drop(columns=["Age"])

# identify categorical column(s)
cat_cols = ["Sex"]
cat_features = [X.columns.get_loc(c) for c in cat_cols]  # CatBoost needs column indices

# -----------------------
# 2) CV Training (MAE)
# -----------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train), dtype=float)

params = dict(
    loss_function="MAE",
    eval_metric="MAE",
    iterations=8000,
    learning_rate=0.03,
    depth=10,
    l2_leaf_reg=4,
    random_seed=42,
    subsample=0.85,
    rsm=0.85,                 # feature subsampling
    min_data_in_leaf=20,
    od_type="Iter",
    od_wait=300,
    verbose=200
)

fold_mae = []
models = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = CatBoostRegressor(**params)
    model.fit(
        X_tr, y_tr,
        cat_features=cat_features,
        eval_set=(X_va, y_va),
        use_best_model=True
    )

    pred_va = model.predict(X_va)
    oof[va_idx] = pred_va

    mae = mean_absolute_error(y_va, pred_va)
    fold_mae.append(mae)
    models.append(model)

    print(f"Fold {fold} MAE: {mae:.6f}")

cv_mae = mean_absolute_error(y, oof)
print("\nFold MAE list:", [round(m, 6) for m in fold_mae])
print("OOF/CV MAE:", round(cv_mae, 6))

# -----------------------
# 3) Train on full data with best_iteration average (optional)
# -----------------------
# (Simple & strong approach: fit one final model with more iterations, early stopping on a small holdout
#  OR just use last model. Here: retrain on full data with same params but less verbose.)

final_params = params.copy()
final_params["verbose"] = 200

final_model = CatBoostRegressor(**final_params)
final_model.fit(X, y, cat_features=cat_features)

# -----------------------
# 4) Predict test & make submission
# -----------------------
test_pred = final_model.predict(test)

submission = pd.DataFrame({
    "id": test["id"],
    "Age": test_pred
})

submission.to_csv("submission2.csv", index=False)
print("\nSaved: submission2.csv")