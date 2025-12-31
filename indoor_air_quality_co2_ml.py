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

import joblib
import pandas as pd
import seaborn as sns
import utils
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import importlib
importlib.reload(utils)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################
base_path = os.path.join(os.getcwd(), "data")
file_name = "Air-Quality-Dataset.csv"
data_path = os.path.join(base_path, file_name)

# Read CSV with semicolon delimiter
df = utils.load_data(data_path)
df.head()

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
    utils.num_summary(df, col, plot=True)

# Correlation analysis between numerical variables
utils.correlation_matrix(df, numeric_cols)

# Correlation summary:
############################
# - Humidity shows weak or negligible correlation with CO2 and particulate matter
# - Temperature is moderately inversely correlated with humidity
# - PM2.5 and PM10 exhibit strong positive correlation (0.66), indicating shared sources
# - Low overall correlation suggests minimal feature redundancy






# Data Preprocessing & Feature Engineering
##############################################################
X_train, X_test, y_train, y_test, scaler = utils.air_quality_pipeline(df)

######################################################
# Base Models
######################################################
preprocessor = 'passthrough'

results = utils.base_models_pipeline(X_train, y_train, preprocessor=preprocessor, scoring="roc_auc", cv=3)

print(results)

cat_cols, num_cols, cat_but_car = utils.grab_col_names(df)
print("Categorical:", cat_cols)
print("Numerical:", num_cols)
print("Cat but cardinal:", cat_but_car)

