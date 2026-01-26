# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# %%
train_path = "train.csv"
test_path = "test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train shape:", train.shape)
print("Test shape :", test.shape)

# %%
TARGET = "Age"
ID_COL = "id"

# %%
# Feature engineering tailored for Abalone-style data

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6
    df["Volume"] = df["Length"] * df["Diameter"] * df["Height"]
    df["Density"] = df["Weight"] / (df["Volume"] + eps)

    df["Shucked_Ratio"] = df["Shucked Weight"] / (df["Weight"] + eps)
    df["Viscera_Ratio"] = df["Viscera Weight"] / (df["Weight"] + eps)
    df["Shell_Ratio"] = df["Shell Weight"] / (df["Weight"] + eps)

    df["Length_Diameter"] = df["Length"] / (df["Diameter"] + eps)
    df["Length_Height"] = df["Length"] / (df["Height"] + eps)
    df["Diameter_Height"] = df["Diameter"] / (df["Height"] + eps)

    df["Weight_Length"] = df["Weight"] / (df["Length"] + eps)
    df["Weight_Diameter"] = df["Weight"] / (df["Diameter"] + eps)
    df["Weight_Height"] = df["Weight"] / (df["Height"] + eps)
    return df

# %%
train_fe = add_features(train)
test_fe = add_features(test)

X = train_fe.drop(columns=[TARGET, ID_COL])
y = train_fe[TARGET]
X_test = test_fe.drop(columns=[ID_COL])

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# %%
# Preprocessor factory (fit only on training fold each time)

def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

# %%
# XGBoost params tuned for MAE
xgb_params = dict(
    objective="reg:absoluteerror",
    tree_method="hist",
    n_estimators=1400,
    learning_rate=0.03,
    max_depth=7,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    gamma=0.0,
    n_jobs=-1,
    random_state=42,
)

# %%
# CV to estimate MAE
kf = KFold(n_splits=5, shuffle=True, random_state=42)
maes = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    pre = build_preprocessor()
    model = XGBRegressor(**xgb_params)
    pipe = Pipeline([("preprocess", pre), ("model", model)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_va)
    mae = mean_absolute_error(y_va, pred)
    maes.append(mae)

    print(f"Fold {fold} MAE: {mae:.6f}")

print(f"CV MAE mean: {np.mean(maes):.6f}")
print(f"CV MAE std : {np.std(maes):.6f}")

# %%
# Train final model on full data
final_pre = build_preprocessor()
final_model = XGBRegressor(**xgb_params)
final_pipe = Pipeline([("preprocess", final_pre), ("model", final_model)])
final_pipe.fit(X, y)
test_pred = final_pipe.predict(X_test)

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    "Age": test_pred
})

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Saved submission to:", out_path)

# %%
submission.head()
