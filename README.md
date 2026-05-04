<div align="center">

# 🐚 Abalone Age Prediction

**Predict the age of abalones from physical measurements using advanced ensemble machine learning**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-enabled-brightgreen)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-enabled-yellow)](https://lightgbm.readthedocs.io/)
[![Gradio](https://img.shields.io/badge/Gradio-App-ff7c00?logo=gradio)](https://huggingface.co/spaces/shovo896/abalone_age-prediction)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/shovo896/abalone_age-prediction)

</div>

---

## 📖 Overview

Determining the age of an abalone traditionally requires cutting the shell and counting rings under a microscope — a tedious and time-consuming process. This project builds a machine learning pipeline to **predict abalone age from easy-to-measure physical attributes**, eliminating the need for manual counting.

The project was developed as a competitive machine learning challenge, with the evaluation metric being **Mean Absolute Error (MAE)**. Multiple modelling strategies — from individual gradient boosted trees to weighted ensembles and stacking regressors — were explored and benchmarked.

---

## 🌐 Live Demo

Try the interactive web app powered by **Gradio** on Hugging Face Spaces:

👉 **[https://huggingface.co/spaces/shovo896/abalone_age-prediction](https://huggingface.co/spaces/shovo896/abalone_age-prediction)**

The app supports:
- **Single Prediction** — enter physical measurements manually and get an instant age estimate.
- **Batch Prediction** — upload a CSV file and download a `submission.csv` with predictions for every row.

---

## 📊 Dataset

The dataset contains physical measurements of abalones. The target variable is **Age** (measured in years).

| Feature | Description |
|---|---|
| `Sex` | M (Male), F (Female), I (Infant) |
| `Length` | Longest shell measurement (mm) |
| `Diameter` | Perpendicular to length (mm) |
| `Height` | Height with meat (mm) |
| `Weight` | Whole abalone weight (g) |
| `Shucked Weight` | Weight of meat (g) |
| `Viscera Weight` | Gut weight after bleeding (g) |
| `Shell Weight` | Weight after drying (g) |
| **`Age`** | **Target — age in years** |

---

## 🧠 Models & Approach

Several modelling strategies were explored:

| Strategy | Models Used | Notes |
|---|---|---|
| **XGBoost + RF Blend** | XGBoost, Random Forest | 5-Fold CV with feature engineering |
| **Weighted Ensemble** | XGBoost, LightGBM, RF, Gradient Boosting | Grid search over blend weights to minimise MAE |
| **Stacking Regressor** | RF, GBR, HistGB, SVR → Ridge meta | Full sklearn pipeline with imputation & scaling |

### Feature Engineering

New features derived from raw measurements to improve model performance:

- **Volume** = Length × Diameter × Height
- **Density** = Weight / Volume
- Weight ratios: `Shucked_Ratio`, `Viscera_Ratio`, `Shell_Ratio`
- Dimension ratios: `Length_Diameter`, `Length_Height`, `Diameter_Height`
- Combined weight-dimension features

---

## 📁 Project Structure

```
abalone_age_prediction/
│
├── 📓 Notebooks
│   ├── model.ipynb          # Baseline modelling notebook
│   ├── model2.ipynb         # Extended experiments
│   ├── cat_rf.ipynb         # CatBoost + Random Forest
│   ├── all_models.ipynb     # Full model comparison
│   ├── random.ipynb         # Random Forest deep-dive
│   └── randommm2.ipynb      # Additional experiments
│
├── 🐍 Training Scripts
│   ├── train_xgb.py         # XGBoost + RF blend with 5-Fold CV
│   ├── ensemble_final.py    # Weighted ensemble (XGB + LGB + RF + GB)
│   ├── optimize_v2.py       # Hyperparameter optimisation v2
│   ├── optimize_extreme.py  # Aggressive hyperparameter search
│   ├── quick_optimize.py    # Fast optimisation pass
│   ├── tune_hyperparams.py  # Hyperparameter tuning utilities
│   ├── model1.py            # Standalone model script
│   └── gmfj.py              # Additional model experiments
│
├── 🌐 Web Application
│   └── app.py               # Gradio web app (stacking ensemble)
│
├── 📂 Data
│   ├── train.csv            # Training data
│   └── test.csv             # Test data
│
├── 📤 Submissions
│   ├── submission.csv
│   ├── submission_ensemble.csv
│   ├── submission_optimized.csv
│   └── ...
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher

### Installation

```bash
# Clone the repository
git clone https://github.com/shovo896/abaleon_age_prediction.git
cd abaleon_age_prediction

# Install dependencies
pip install -r requirements.txt
```

### Run the Web App Locally

```bash
python app.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

### Train the Ensemble Model

```bash
# XGBoost + Random Forest blend (5-Fold CV)
python train_xgb.py

# Optimised weighted ensemble (XGB + LGB + RF + GB)
python ensemble_final.py
```

---

## 📦 Dependencies

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
gradio>=4.0
xgboost
lightgbm
```

---

## 📈 Results

Models were evaluated using **Mean Absolute Error (MAE)** on a held-out validation set. Lower is better.

| Model | Validation MAE |
|---|---|
| XGBoost (single) | — |
| LightGBM (single) | — |
| Random Forest | — |
| **Weighted Ensemble (XGB + LGB + RF + GB)** | **Best** |
| Stacking Regressor | Competitive |

> Exact scores depend on the random seed and data split. See notebook outputs for detailed results.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

<div align="center">

Made with ❤️ by [shovo896](https://github.com/shovo896)

</div>
