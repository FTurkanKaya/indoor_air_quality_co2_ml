################################################
# End-to-End Diabetes Machine Learning Pipeline I
################################################

# 1. Data Loading
# 2. Data Validation (sensor sanity checks)
# 3. Exploratory Data Analysis (EDA)
# 4. Domain-based Cleaning
# 5. Feature Engineering (time + interaction)
# 6. Train / Test Split (time-aware)
# 7. Baseline Regression
# 8. Advanced Models
# 9. Evaluation & Interpretation
#################################################

import os
sys.path.append(os.path.join(os.getcwd(), "src"))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# Import functions from src modules
from data_loader import load_data
from eda import num_summary, cat_summary, correlation_matrix
from cleaning import replace_with_thresholds, clean_data, outlier_thresholds
from features import add_time_features, add_interaction_features, grab_col_names, one_hot_encoder
from models import build_regression_models, train_models, preprocess_for_model, regression_hyperparameter_optimization
from evaluation import regression_metrics, plot_predictions, feature_importance

# =============================================
# 1. LOAD DATA
# =============================================
base_path = os.path.join(os.getcwd(), "data")
file_name = "Air-Quality-Dataset.csv"
data_path = os.path.join(base_path, file_name)
df = load_data(data_path)


################################################
# 2. Exploratory Data Analysis
################################################
# --- Data Overview ---
df.info()       # Check null values and types
df.describe().T   # Summary statistics for numeric columns

# Dataset Summary (n = 6200)
# No missing values detected across variables.

# CO2:
# Mean (~432 ppm) and median (400 ppm) are close, indicating a relatively balanced distribution.
# Maximum value (3000 ppm) is an extreme outlier and may indicate poor air quality or sensor anomaly.
# Values >2000 ppm should be reviewed.

# PM2.5:
# Right-skewed distribution (mean > median) with very high variance.
# Invalid values detected (e.g., -1).
# Extreme maximum (999.9) likely represents sensor saturation or error code.
# Requires strict filtering and outlier handling.

# PM10:
# Strong right skew and large variability.
# Physically invalid minimum (-1) and extreme maximum (1999.9).
# Similar preprocessing strategy to PM2.5 is required.

# Temperature:
# Stable and realistic range (11–21.2 °C).
# Low variance indicates reliable sensor measurements.
# No immediate preprocessing required.

# Humidity:
# Values exceed expected 0–100% range.
# Likely incorrect scaling or non-standard sensor unit.
# Must be rescaled or reinterpreted before use.

# Status:
# Binary variable (0/1) with ~9.8% positive class.
# Indicates class imbalance.
# Accuracy alone is insufficient; use precision/recall/F1 for evaluation.


# --- Check unique values for categorical columns ---
df['CO2 CATEGORY'].value_counts()
df['PM2.5 CATEGORY'].value_counts()
df['PM10 CATEGORY'].value_counts()


# --- Time series plot for CO2 ---
plt.figure()
plt.plot(df["TIME"], df["CO2"])
plt.title("CO2 Concentration Over Time")
plt.xlabel("Time")
plt.ylabel("CO2 (ppm)")
plt.show()

# --- Distribution plots for numeric features ---
numeric_cols = ["CO2", "PM2.5", "PM10", "TEMPERATURE", "HUMIDITY"]

for col in numeric_cols:
    plt.figure()
    plt.hist(df[col], bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


# Visualization of numerical variables
for col in numeric_cols:
    num_summary(df, col, plot=True)

# Correlation analysis between numerical variables
correlation_matrix(df, numeric_cols)

# Correlation summary:
############################
# - Humidity shows weak or negligible correlation with CO2 and particulate matter
# - Temperature is moderately inversely correlated with humidity
# - PM2.5 and PM10 exhibit strong positive correlation (0.66), indicating shared sources
# - Low overall correlation suggests minimal feature redundancy


# =============================================
# 3. DATA CLEANING
# =============================================
# Clip negative values for numeric columns
df = clean_data(df, cols_to_clip=numeric_cols, time_col="TIME")

# Replace outliers with thresholds
for col in numeric_cols:
    replace_with_thresholds(df, col)


print(df.info())
print(df.describe().T)  # Min, Max, Mean, 25/50/75% quantiles, vs.
print(df.isnull().sum())

# ==========================================================
# Observations:
# ==========================================================
# 1. Missing values: All columns have 0 missing values
# 2. Data types:
#    - TIME: datetime64[ns, UTC]
#    - CO2, Status: int64
#    - PM2.5, PM10, TEMPERATURE, HUMIDITY: float64
#    - Categorical columns: object
# 3. Outliers:
#    - CO2: max 498 (previous outlier 3000 removed)
#    - PM2.5: max 15.55 (previous extreme 999.9 removed)
#    - PM10: max 38.55 (previous extreme 1999.9 removed)
#    - HUMIDITY: still very high (~622), requires further correction



# ==========================================================
# Feature Engineering
# ==========================================================

# 1. Time-based Features
# - Extract hour, day of week, and weekend indicator from timestamp
df = add_time_features(df, time_col="TIME")
df.head()

# 2. Interaction Features
# - Create new features by combining existing numeric variables
#   a) temp_humidity: interaction between TEMPERATURE and HUMIDITY
#   b) pm_total: sum of PM2.5 and PM10 to represent total particulate matter
df = add_interaction_features(df)

# 3. Column Summary
# - Identify categorical columns, numeric columns, and categorical but cardinal columns
cat_cols, num_cols, cat_but_car = grab_col_names(df)
print("Categorical Columns:", cat_cols)
print("Numerical Columns:", num_cols)
print("Categorical but Cardinal Columns:", cat_but_car)

# ==============================
# ONE-HOT ENCODING
# ==============================

df_encoded = one_hot_encoder(df, categorical_cols=cat_cols)

print(df_encoded.columns)


# ==========================================================
# HUMIDITY Normalization / Scaling
# ==========================================================

# - HUMIDITY values are unusually high (~622)
# - We can scale them to 0-100% range using MinMaxScaler or simple division
df["HUMIDITY"] = (df["HUMIDITY"] - df["HUMIDITY"].min()) / (df["HUMIDITY"].max() - df["HUMIDITY"].min()) * 100

# Verify the scaled HUMIDITY
print(df["HUMIDITY"].describe())

# ==========================================================
# Observations:
# ==========================================================
# 1. Time features added for potential temporal patterns
# 2. Interaction features help models capture combined effects
# 3. HUMIDITY normalized to 0-100 range to make it physically meaningful
# 4. Dataset is now ready for model training


# ========================================
# Preprocessing & Train/Test Split
# ========================================

# Here we scale numerical features and split the dataset into training and testing sets
X_train, X_test, y_train, y_test, scaler = preprocess_for_model(df_encoded,
                                                                target_col="CO2",
                                                                scaler_type="standard",
                                                                test_size=0.2,
                                                                random_state=42)

X_train.columns
# ========================================
# 2. Build Regression Models
# ========================================
# Create a dictionary of regression models to train
models = build_regression_models()

# ========================================
# 3. Train Models
# ========================================
# Fit each model on the training data
trained_models = train_models(models, X_train, y_train)

# ========================================
# 4. Evaluate Models
# ========================================
# Evaluate each model on the test set
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    print(f"{name} performance:")
    print(metrics)
    print("-"*40)

# ========================================
# 5. Visualize Predictions
# ========================================
# Plot actual vs predicted CO2 for one of the models (e.g., RandomForest)
plot_predictions(y_test, trained_models["RandomForest"].predict(X_test))

# ========================================
# 6. Feature Importance
# ========================================
# Display top features for tree-based models
feature_importance(trained_models["RandomForest"], X_train.columns)
