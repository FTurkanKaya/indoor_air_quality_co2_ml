################################################
# End-to-End Diabetes Machine Learning Pipeline I
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

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
# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################
base_path = os.path.join(os.getcwd(), "data")
file_name = "Air-Quality-Dataset.csv"
data_path = os.path.join(base_path, file_name)

df = utils.load_data(data_path)
# --- Check first rows ---
df.head()

df.dtypes



# Read CSV with semicolon delimiter
df = pd.read_csv("data/Air-Quality-Dataset.csv", sep=';', decimal='.')

# Convert TIME column to datetime
df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce')

# Check dtypes
df.dtypes

# --- Data Overview ---
df.info()       # Check null values and types
df.describe()   # Summary statistics for numeric columns

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
