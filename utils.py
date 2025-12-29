################################################
# IMPORTS
################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor

################################################
# 1. DATA LOADING & BASIC HELPERS
################################################

def load_data(filepath):
    """
    Load CSV file with semicolon separator and decimal point handling,
    convert timestamp column to datetime, sort by timestamp.
    """
    # Read CSV with proper separator and decimal point
    df = pd.read_csv(
        filepath,
        sep=';',  # Semicolon separator
    )

    # Convert timestamp column if exists
    if "TIME" in df.columns:
        df["TIME"] = pd.to_datetime(df["TIME"], errors='coerce')
        df.sort_values("TIME", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def clean_data(df, cols_to_clip=None):
    """Eksik veya negatif değerleri temizler, interpolasyon uygular."""
    df = df.copy()
    # Negatif değerleri sıfırla
    if cols_to_clip:
        for col in cols_to_clip:
            df[col] = df[col].clip(lower=0)
    # Eksik değerleri zaman bazlı interpolate ile doldur
    df.interpolate(method='time', inplace=True)
    df.dropna(inplace=True)
    return df

################################################
# 2. FEATURE ENGINEERING
################################################
def add_time_features(df):
    """Timestamp bazlı özellikler ekler."""
    df = df.copy()
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        df["hour"] = ts.dt.hour
        df["dayofweek"] = ts.dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    return df

def add_interaction_features(df):
    """Çevresel parametreler arasında etkileşim özellikleri ekler."""
    df = df.copy()
    if {"Temperature","Humidity"}.issubset(df.columns):
        df["temp_humidity"] = df["Temperature"] * df["Humidity"]
    if {"PM2.5","PM10"}.issubset(df.columns):
        df["pm_total"] = df["PM2.5"] + df["PM10"]
    return df

################################################
# 3. OUTLIER HANDLING
################################################
def outlier_thresholds(df, col, q1=0.25, q3=0.75):
    """IQR yöntemi ile aykırı değer sınırlarını döndürür."""
    q_low = df[col].quantile(q1)
    q_high = df[col].quantile(q3)
    iqr = q_high - q_low
    return q_low - 1.5*iqr, q_high + 1.5*iqr

def replace_outliers(df, col):
    """Aykırı değerleri IQR sınırlarıyla değiştirir."""
    df = df.copy()
    low, high = outlier_thresholds(df, col)
    df[col] = np.where(df[col] < low, low, df[col])
    df[col] = np.where(df[col] > high, high, df[col])
    return df

################################################
# 4. SCALING
################################################
def scale_features(df, numeric_cols, scaler_type="standard"):
    """Sayısal kolonları ölçeklendirir."""
    df = df.copy()
    if scaler_type=="standard":
        scaler = StandardScaler()
    elif scaler_type=="minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type 'standard' veya 'minmax' olmalı")
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler

################################################
# 5. TRAIN-TEST SPLIT
################################################
def split_data(df, feature_cols, target_col, test_size=0.2, random_state=42):
    """Train-test split yapar."""
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

################################################
# 6. REGRESSION MODELS
################################################
def build_regression_models():
    """Örnek regresyon modelleri ve parametreleri döner."""
    regressors = {
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=200),
        "XGBoost": XGBRegressor(eval_metric='rmse'),
        "LightGBM": LGBMRegressor()
    }
    return regressors

def hyperparameter_optimization(X_train, y_train, X_test, y_test, regressors, cv=3, scoring="r2"):
    """GridSearchCV ile regresyon modellerini optimize eder ve test skorunu döner."""
    best_models = {}
    scores = {}
    for name, model in regressors.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        best_models[name] = model
        scores[name] = score
        print(f"{name} {scoring} score: {score:.4f}")
    return best_models, scores

################################################
# 7. PLOTTING
################################################
def plot_feature_distribution(df, cols):
    """Sayısal kolonların dağılımını histogram ve boxplot ile gösterir."""
    for col in cols:
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f"{col} Histogram")
        plt.subplot(1,2,2)
        sns.boxplot(x=df[col])
        plt.title(f"{col} Boxplot")
        plt.show()


