# %% [markdown]
# # Abalone Age Prediction - Stacking with Random Forest & XGBoost

# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# %% [markdown]
# ## 1. Load Data

# %%
# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nFirst few rows:")
print(train_df.head())

# %%
# Basic statistics
print("\nTarget variable statistics:")
print(train_df['Age'].describe())
print("\nMissing values:")
print(train_df.isnull().sum())

# %% [markdown]
# ## 2. Feature Engineering

# %%
def feature_engineering(df):
    """
    Create new features to improve model performance
    """
    df = df.copy()
    
    # Ratio features
    df['Length_to_Diameter'] = df['Length'] / (df['Diameter'] + 1e-5)
    df['Height_to_Diameter'] = df['Height'] / (df['Diameter'] + 1e-5)
    df['Height_to_Length'] = df['Height'] / (df['Length'] + 1e-5)
    
    # Volume approximation (cylinder)
    df['Volume'] = np.pi * (df['Diameter']/2)**2 * df['Height']
    
    # Density features
    df['Density'] = df['Weight'] / (df['Volume'] + 1e-5)
    df['Shell_Density'] = df['Shell Weight'] / (df['Volume'] + 1e-5)
    
    # Weight ratios
    df['Shucked_to_Total'] = df['Shucked Weight'] / (df['Weight'] + 1e-5)
    df['Viscera_to_Total'] = df['Viscera Weight'] / (df['Weight'] + 1e-5)
    df['Shell_to_Total'] = df['Shell Weight'] / (df['Weight'] + 1e-5)
    
    # Total meat weight
    df['Total_Meat'] = df['Shucked Weight'] + df['Viscera Weight']
    df['Meat_to_Shell'] = df['Total_Meat'] / (df['Shell Weight'] + 1e-5)
    
    # Surface area approximation
    df['Surface_Area'] = 2 * np.pi * (df['Diameter']/2) * (df['Height'] + df['Diameter']/2)
    
    # Weight per unit length
    df['Weight_per_Length'] = df['Weight'] / (df['Length'] + 1e-5)
    
    # Polynomial features
    df['Length_squared'] = df['Length'] ** 2
    df['Diameter_squared'] = df['Diameter'] ** 2
    df['Weight_squared'] = df['Weight'] ** 2
    
    # Interaction features
    df['Length_x_Diameter'] = df['Length'] * df['Diameter']
    df['Length_x_Height'] = df['Length'] * df['Height']
    
    return df

# Apply feature engineering
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

print("New feature count:", train_df.shape[1])

# %% [markdown]
# ## 3. Data Preprocessing

# %%
# Encode categorical variable
le = LabelEncoder()
train_df['Sex_encoded'] = le.fit_transform(train_df['Sex'])
test_df['Sex_encoded'] = le.transform(test_df['Sex'])

# Create one-hot encoding for Sex
train_sex_dummies = pd.get_dummies(train_df['Sex'], prefix='Sex')
test_sex_dummies = pd.get_dummies(test_df['Sex'], prefix='Sex')

train_df = pd.concat([train_df, train_sex_dummies], axis=1)
test_df = pd.concat([test_df, test_sex_dummies], axis=1)

# Prepare features and target
drop_cols = ['id', 'Sex', 'Age']
feature_cols = [col for col in train_df.columns if col not in drop_cols]

X = train_df[feature_cols]
y = train_df['Age']
X_test = test_df[feature_cols]
test_ids = test_df['id']

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# %% [markdown]
# ## 4. Stacking Model Setup

# %%
# Cross-validation setup
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

# Base Models
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=SEED,
    n_jobs=-1
)

xgb_model = xgb.XGBRegressor(
    objective='reg:absoluteerror',
    eval_metric='mae',
    max_depth=7,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=SEED,
    verbosity=0
)

# Meta-learner (Stacking model)
meta_model = Ridge(alpha=1.0)

print("Base Models:")
print("1. Random Forest")
print("2. XGBoost")
print("\nMeta-learner: Ridge Regression")

# %% [markdown]
# ## 5. Training Base Models with Out-of-Fold Predictions

# %%
# Initialize arrays for out-of-fold predictions
rf_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))

# Initialize arrays for test predictions
rf_test_preds = np.zeros(len(X_test))
xgb_test_preds = np.zeros(len(X_test))

print("Training Base Models with Cross-Validation...\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"Fold {fold}/{n_folds}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train Random Forest
    rf_model.fit(X_train, y_train)
    rf_oof[val_idx] = rf_model.predict(X_val)
    rf_test_preds += rf_model.predict(X_test) / n_folds
    
    rf_mae = mean_absolute_error(y_val, rf_oof[val_idx])
    print(f"  Random Forest MAE: {rf_mae:.4f}")
    
    # Train XGBoost
    xgb_model.fit(X_train, y_train)
    xgb_oof[val_idx] = xgb_model.predict(X_val)
    xgb_test_preds += xgb_model.predict(X_test) / n_folds
    
    xgb_mae = mean_absolute_error(y_val, xgb_oof[val_idx])
    print(f"  XGBoost MAE: {xgb_mae:.4f}\n")

# Calculate overall OOF scores for base models
rf_overall_mae = mean_absolute_error(y, rf_oof)
xgb_overall_mae = mean_absolute_error(y, xgb_oof)

print("="*60)
print("BASE MODEL PERFORMANCE (Out-of-Fold)")
print("="*60)
print(f"Random Forest MAE: {rf_overall_mae:.4f}")
print(f"XGBoost MAE: {xgb_overall_mae:.4f}")
print("="*60)

# %% [markdown]
# ## 6. Training Meta-Learner (Stacking)

# %%
# Create training data for meta-learner
meta_train = np.column_stack([rf_oof, xgb_oof])
meta_test = np.column_stack([rf_test_preds, xgb_test_preds])

print("\nTraining Meta-Learner (Ridge Regression)...")
meta_model.fit(meta_train, y)

# Make predictions using stacked model
stacked_oof = meta_model.predict(meta_train)
stacked_test_pred = meta_model.predict(meta_test)

# Calculate stacking performance
stacked_mae = mean_absolute_error(y, stacked_oof)

print("\n" + "="*60)
print("STACKING MODEL PERFORMANCE")
print("="*60)
print(f"Stacked Model MAE: {stacked_mae:.4f}")
print("="*60)

# Check meta-learner weights
print(f"\nMeta-learner coefficients:")
print(f"  Random Forest weight: {meta_model.coef_[0]:.4f}")
print(f"  XGBoost weight: {meta_model.coef_[1]:.4f}")
print(f"  Intercept: {meta_model.intercept_:.4f}")

# %% [markdown]
# ## 7. Model Comparison

# %%
# Compare all approaches
results_df = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'Stacked (RF + XGB)'],
    'MAE': [rf_overall_mae, xgb_overall_mae, stacked_mae]
})

results_df = results_df.sort_values('MAE')
print("\n\nModel Performance Comparison:")
print(results_df.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = plt.barh(results_df['Model'], results_df['MAE'], color=colors)
plt.xlabel('Mean Absolute Error (MAE)')
plt.title('Model Performance Comparison')
plt.axvline(x=1.26, color='orange', linestyle='--', linewidth=2, label='Target MAE = 1.26')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate improvement
best_base_mae = min(rf_overall_mae, xgb_overall_mae)
improvement = ((best_base_mae - stacked_mae) / best_base_mae) * 100
print(f"\nImprovement from stacking: {improvement:.2f}%")

# %% [markdown]
# ## 8. Feature Importance Analysis

# %%
# Retrain final models on full data to get feature importance
rf_final = RandomForestRegressor(
    n_estimators=500, max_depth=25, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt', random_state=SEED, n_jobs=-1
)
rf_final.fit(X, y)

xgb_final = xgb.XGBRegressor(
    objective='reg:absoluteerror', eval_metric='mae', max_depth=7,
    learning_rate=0.05, n_estimators=1000, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1,
    reg_lambda=0.1, random_state=SEED, verbosity=0
)
xgb_final.fit(X, y)

# Random Forest Feature Importance
rf_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_final.feature_importances_
}).sort_values('importance', ascending=False).head(15)

# XGBoost Feature Importance
xgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_final.feature_importances_
}).sort_values('importance', ascending=False).head(15)

# Plot feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].barh(rf_importance['feature'], rf_importance['importance'])
axes[0].set_xlabel('Importance')
axes[0].set_title('Top 15 Features - Random Forest')
axes[0].invert_yaxis()

axes[1].barh(xgb_importance['feature'], xgb_importance['importance'])
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 15 Features - XGBoost')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

print("\nTop 10 Features - Random Forest:")
print(rf_importance.head(10).to_string(index=False))
print("\n\nTop 10 Features - XGBoost:")
print(xgb_importance.head(10).to_string(index=False))

# %% [markdown]
# ## 9. Prediction Analysis

# %%
# Analyze predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Actual vs Predicted
axes[0].scatter(y, stacked_oof, alpha=0.3, s=10)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Age')
axes[0].set_ylabel('Predicted Age')
axes[0].set_title(f'Actual vs Predicted\nMAE: {stacked_mae:.4f}')
axes[0].grid(True, alpha=0.3)

# Residual Plot
residuals = y - stacked_oof
axes[1].scatter(stacked_oof, residuals, alpha=0.3, s=10)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Age')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')
axes[1].grid(True, alpha=0.3)

# Residual Distribution
axes[2].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[2].axvline(x=0, color='r', linestyle='--', lw=2)
axes[2].set_xlabel('Residuals')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Residual Distribution')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nPrediction Statistics:")
print(f"Mean Residual: {residuals.mean():.4f}")
print(f"Std Residual: {residuals.std():.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, stacked_oof)):.4f}")
print(f"R² Score: {r2_score(y, stacked_oof):.4f}")

# %% [markdown]
# ## 10. Create Submission File

# %%
# Create submission dataframe
submission = pd.DataFrame({
    'id': test_ids,
    'Age': stacked_test_pred
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("✓ Submission file created: submission.csv")
print(f"\nSubmission shape: {submission.shape}")
print("\nFirst few predictions:")
print(submission.head(10))
print("\nPrediction statistics:")
print(submission['Age'].describe())

# %% [markdown]
# ## 11. Final Summary

# %%
print("\n" + "="*70)
print(" "*20 + "FINAL RESULTS SUMMARY")
print("="*70)
print(f"\nTarget MAE: 1.26")
print(f"Achieved MAE (Stacking): {stacked_mae:.4f}")
print(f"\nBase Model Performance:")
print(f"  Random Forest MAE: {rf_overall_mae:.4f}")
print(f"  XGBoost MAE: {xgb_overall_mae:.4f}")
print(f"\nStacking Improvement: {improvement:.2f}%")
print(f"\nMeta-learner Weights:")
print(f"  Random Forest: {meta_model.coef_[0]:.4f}")
print(f"  XGBoost: {meta_model.coef_[1]:.4f}")
print(f"\nTotal Features Used: {len(feature_cols)}")
print(f"Training Samples: {len(X)}")
print(f"Test Samples: {len(X_test)}")
print("\n✓ Submission file ready: submission.csv")
print("="*70)