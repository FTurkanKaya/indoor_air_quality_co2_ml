import pandas as pd

# Time-based features
def add_time_features(df, time_col="TIME"):
    df = df.copy()
    if time_col in df.columns:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        df["hour"] = ts.dt.hour
        df["dayofweek"] = ts.dt.dayofweek
        df["is_weekend"] = ts.dt.dayofweek.isin([5,6]).astype(int)
    return df

# Interaction features
def add_interaction_features(df):
    df = df.copy()
    if {"TEMPERATURE","HUMIDITY"}.issubset(df.columns):
        df["temp_humidity"] = df["TEMPERATURE"] * df["HUMIDITY"]
    if {"PM2.5","PM10"}.issubset(df.columns):
        df["pm_total"] = df["PM2.5"] + df["PM10"]
    return df

# One-hot encoding
def one_hot_encoder(df, categorical_cols, drop_first=True):
    return pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)

# Column grabber
def grab_col_names(df, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtype == "O"]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtype != "O"]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and df[col].dtype == "O"]
    cat_cols = [col for col in cat_cols + num_but_cat if col not in cat_but_car]
    num_cols = [col for col in df.columns if df[col].dtype != "O" and col not in num_but_cat]
    if "TIME" in num_cols: num_cols.remove("TIME")
    if "TIME" in cat_cols: cat_cols.remove("TIME")
    return cat_cols, num_cols, cat_but_car

