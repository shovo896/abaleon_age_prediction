from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import pandas as pd
import gradio as gr

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor

DATA_DIR = os.getenv("DATA_DIR", ".")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUTPUT_COL = "age"

FEATURE_DEFAULTS = {
    "Sex": "I",
    "Length": 0.5,
    "Diameter": 0.4,
    "Height": 0.1,
    "Weight": 0.8,
    "Shucked Weight": 0.3,
    "Viscera Weight": 0.15,
    "Shell Weight": 0.2,
}


def _load_train() -> pd.DataFrame:
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"train.csv not found at {TRAIN_PATH}")
    return pd.read_csv(TRAIN_PATH)


def _load_test_optional() -> pd.DataFrame | None:
    if os.path.exists(TEST_PATH):
        return pd.read_csv(TEST_PATH)
    return None


def _detect_target(train: pd.DataFrame, test: pd.DataFrame | None) -> str:
    if test is not None:
        diff = list(set(train.columns) - set(test.columns))
        if len(diff) == 1:
            return diff[0]
    # Fallbacks
    if "Age" in train.columns:
        return "Age"
    if "age" in train.columns:
        return "age"
    return train.columns[-1]


def _build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )
    return preprocess


@lru_cache(maxsize=1)
def get_model() -> Tuple[Pipeline, List[str]]:
    train = _load_train()
    test = _load_test_optional()

    target_col = _detect_target(train, test)
    id_col = "id" if "id" in train.columns else None

    X = train.drop(columns=[target_col], errors="ignore")
    y = train[target_col]

    if id_col:
        X = X.drop(columns=[id_col], errors="ignore")

    preprocess = _build_preprocess(X)

    n_jobs = int(os.getenv("N_JOBS", "-1"))

    base_models = [
        ("rf", RandomForestRegressor(
            n_estimators=1200, random_state=42, n_jobs=n_jobs
        )),
        ("gbr", GradientBoostingRegressor(random_state=42)),
        ("hgb", HistGradientBoostingRegressor(random_state=42)),
        ("svr", SVR(C=10, epsilon=0.1))
    ]

    final_model = Ridge(alpha=1.0, random_state=42)

    stack = StackingRegressor(
        estimators=base_models,
        final_estimator=final_model,
        passthrough=False,
        n_jobs=n_jobs
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("stack", stack)
    ])

    model.fit(X, y)
    return model, list(X.columns)


def _align_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_cols]


def predict_single(sex, length, diameter, height, weight, shucked_weight, viscera_weight, shell_weight):
    model, feature_cols = get_model()
    row = {
        "Sex": sex,
        "Length": length,
        "Diameter": diameter,
        "Height": height,
        "Weight": weight,
        "Shucked Weight": shucked_weight,
        "Viscera Weight": viscera_weight,
        "Shell Weight": shell_weight,
    }
    df = pd.DataFrame([row])
    df = _align_features(df, feature_cols)
    pred = model.predict(df)[0]
    return float(pred)


def predict_csv(file_obj):
    if file_obj is None:
        return None, pd.DataFrame({"error": ["Please upload a CSV file."]})

    model, feature_cols = get_model()

    df = pd.read_csv(file_obj.name)
    id_col = "id" if "id" in df.columns else None

    features = df.drop(columns=[id_col], errors="ignore")
    features = _align_features(features, feature_cols)

    preds = model.predict(features)

    if id_col is None:
        out = pd.DataFrame({"id": np.arange(len(df)), OUTPUT_COL: preds})
    else:
        out = pd.DataFrame({id_col: df[id_col], OUTPUT_COL: preds})

    out_path = "submission.csv"
    out.to_csv(out_path, index=False)

    return out_path, out.head(10)


with gr.Blocks(title="Abalone Age Prediction") as demo:
    gr.Markdown(
        "# Abalone Age Prediction\n"
        "This app trains the notebook's stacking model on `train.csv` and predicts abalone age. "
        "Upload a CSV for batch predictions or enter a single sample."
    )

    with gr.Tab("Single Prediction"):
        with gr.Row():
            sex = gr.Dropdown(choices=["M", "F", "I"], value=FEATURE_DEFAULTS["Sex"], label="Sex")
            length = gr.Number(value=FEATURE_DEFAULTS["Length"], label="Length")
            diameter = gr.Number(value=FEATURE_DEFAULTS["Diameter"], label="Diameter")
            height = gr.Number(value=FEATURE_DEFAULTS["Height"], label="Height")
        with gr.Row():
            weight = gr.Number(value=FEATURE_DEFAULTS["Weight"], label="Weight")
            shucked_weight = gr.Number(value=FEATURE_DEFAULTS["Shucked Weight"], label="Shucked Weight")
            viscera_weight = gr.Number(value=FEATURE_DEFAULTS["Viscera Weight"], label="Viscera Weight")
            shell_weight = gr.Number(value=FEATURE_DEFAULTS["Shell Weight"], label="Shell Weight")

        predict_btn = gr.Button("Predict")
        pred_output = gr.Number(label="Predicted Age")

        predict_btn.click(
            fn=predict_single,
            inputs=[sex, length, diameter, height, weight, shucked_weight, viscera_weight, shell_weight],
            outputs=pred_output
        )

    with gr.Tab("Batch Prediction"):
        gr.Markdown(
            "Upload a CSV with the same feature columns as `train.csv` (the `id` column is optional)."
        )
        file_input = gr.File(file_types=[".csv"], label="CSV file")
        batch_btn = gr.Button("Generate submission")
        file_output = gr.File(label="Download predictions (submission.csv)")
        preview = gr.Dataframe(label="Preview (first 10 rows)")

        batch_btn.click(
            fn=predict_csv,
            inputs=[file_input],
            outputs=[file_output, preview]
        )


if __name__ == "__main__":
    demo.launch()
