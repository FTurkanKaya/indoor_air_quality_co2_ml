################################################
# End-to-End Indoor Air Quality Regression Pipeline
################################################

# 1. Data Loading
# 2. Data Validation (sensor sanity checks)
# 3. Exploratory Data Analysis (EDA)
# 4. Domain-based Cleaning
# 5. Feature Engineering (time + interaction)
# 6. Train / Test Split
# 7. Baseline Regression
# 8. Advanced Models
# 9. Evaluation & Interpretation
#################################################

# 1. Import Libraries
import os
import sys
import pandas as pd
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Add src folder to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Import custom modules
from data_loader import load_data
from eda import num_summary, cat_summary, correlation_matrix
from cleaning import replace_with_thresholds, clean_data, outlier_thresholds
from features import add_time_features, add_interaction_features, grab_col_names, one_hot_encoder
from models import build_regression_models, train_models, preprocess_for_model, regression_hyperparameter_optimization, cross_validate_models
from evaluation import regression_metrics, plot_predictions, feature_importance

# =============================================
# 2. Load Data
# =============================================
base_path = os.path.join(os.getcwd(), "data")
file_name = "Air-Quality-Dataset.csv"
data_path = os.path.join(base_path, file_name)
df = load_data(data_path)

# =============================================
# 3. Exploratory Data Analysis
# =============================================
df.info()       # Check column types and null values
df.describe().T # Summary statistics

# Distribution plots for numeric features
numeric_cols = ["CO2", "PM2.5", "PM10", "TEMPERATURE", "HUMIDITY"]

for col in numeric_cols:
    plt.figure()
    plt.hist(df[col], bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Correlation analysis
correlation_matrix(df, numeric_cols)

# =============================================
# 4. Data Cleaning
# =============================================
# Clip negative values and interpolate missing data
df = clean_data(df, cols_to_clip=numeric_cols, time_col="TIME")

# Replace extreme outliers with thresholds
for col in numeric_cols:
    replace_with_thresholds(df, col)

# Verify cleaning
print(df.info())
print(df.describe().T)
print(df.isnull().sum())

# =============================================
# 5. Feature Engineering
# =============================================
# 5.1 Time-based features
df = add_time_features(df, time_col="TIME")

# 5.2 Interaction features
df = add_interaction_features(df)

# 5.3 Identify column types
cat_cols, num_cols, cat_but_car = grab_col_names(df)
print("Categorical Columns:", cat_cols)
print("Numerical Columns:", num_cols)
print("Categorical but Cardinal Columns:", cat_but_car)

# 5.4 One-hot encode categorical variables
df_encoded = one_hot_encoder(df, categorical_cols=cat_cols)

# =============================================
# 6. Train/Test Split
# =============================================
X_train, X_test, y_train, y_test, scaler = preprocess_for_model(
    df_encoded,
    target_col="CO2",
    scaler_type="standard",
    test_size=0.2,
    random_state=42
)

# =============================================
# 7. Build Regression Models
# =============================================
models = build_regression_models()

# =============================================
# 8. Cross-Validation
# =============================================
cv_results = cross_validate_models(models, X_train, y_train, cv=5)

# =============================================
# 9. Hyperparameter Optimization
# =============================================
# Define hyperparameter grids
param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

param_grid_xgb = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}

param_grid_lgbm = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}

# Instantiate model objects
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(eval_metric='rmse', random_state=42)
lgbm = LGBMRegressor(random_state=42)

# Apply GridSearchCV
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_lgbm = GridSearchCV(lgbm, param_grid_lgbm, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)

# Fit models
grid_rf.fit(X_train, y_train)
grid_xgb.fit(X_train, y_train)
grid_lgbm.fit(X_train, y_train)

# Print best hyperparameters and CV scores
print("RandomForest best params:", grid_rf.best_params_)
print("RandomForest best CV RMSE:", -grid_rf.best_score_)

print("XGBoost best params:", grid_xgb.best_params_)
print("XGBoost best CV RMSE:", -grid_xgb.best_score_)

print("LightGBM best params:", grid_lgbm.best_params_)
print("LightGBM best CV RMSE:", -grid_lgbm.best_score_)


# =============================================
# 10. Evaluate on Test Set
# =============================================
y_pred_rf = grid_rf.predict(X_test)
y_pred_xgb = grid_xgb.predict(X_test)
y_pred_lgbm = grid_lgbm.predict(X_test)

print("RandomForest test metrics:", regression_metrics(y_test, y_pred_rf))
print("XGBoost test metrics:", regression_metrics(y_test, y_pred_xgb))
print("LightGBM test metrics:", regression_metrics(y_test, y_pred_lgbm))

# Save best RandomForest model
model_dir = os.path.join(os.getcwd(), "models")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(
    grid_rf.best_estimator_,
    os.path.join(model_dir, "random_forest_best.pkl")
)

# =============================================
# 11. Visualize Predictions
# =============================================
plot_predictions(y_test, y_pred_rf, title="RandomForest Predictions")

# =============================================
# 12. Feature Importance
# =============================================
# Create trained_models dictionary to pass into feature importance
trained_models = {
    "RandomForest": grid_rf.best_estimator_,
    "XGBoost": grid_xgb.best_estimator_,
    "LightGBM": grid_lgbm.best_estimator_
}

# Feature importance for RandomForest
best_rf = grid_rf.best_estimator_

feature_importance(best_rf, X_train.columns)


# =============================================
# Test Best Model – Predictions & Interpretation
# =============================================
# 1. Predict on test set
y_pred_test = best_rf.predict(X_test)

# 2. Display first few predictions vs actual values
print("First 10 predictions vs actual:")
for actual, pred in zip(y_test[:10], y_pred_test[:10]):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

# 4. Compute test metrics
test_metrics = regression_metrics(y_test, y_pred_test)
print("\nRandomForest Test Metrics:", test_metrics)

# 5. Plot predictions vs actual
plot_predictions(y_test, y_pred_test, title="RandomForest Test Predictions")

# 6. Feature importance
feature_importance(best_rf, X_train.columns)

