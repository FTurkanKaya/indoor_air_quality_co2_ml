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


# Data Preprocessing & Feature Engineering
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # Categorical columns
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    # Numerical but categorical
    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]

    # Categorical but cardinal
    cat_but_car = [
        col for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Numerical columns (exclude datetime)
    num_cols = [
        col for col in dataframe.columns
        if dataframe[col].dtypes != "O"
        and not pd.api.types.is_datetime64_any_dtype(dataframe[col])
    ]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car
################################################
# Exploratory Data Analysis
################################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)



def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)



def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")



def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")



def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

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
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

################################################
# 1. OUTLIER FUNCTIONS
################################################
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """Calculate lower and upper limits for outliers using IQR."""
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

def replace_with_thresholds(dataframe, col_name):
    """Replace outliers with thresholds."""
    low, up = outlier_thresholds(dataframe, col_name)
    dataframe.loc[dataframe[col_name] < low, col_name] = low
    dataframe.loc[dataframe[col_name] > up, col_name] = up

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """One-hot encode categorical variables."""
    return pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

################################################
# 2. COLUMN GRAB FUNCTION
################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """Get categorical, numerical, and categorical-but-cardinal column names."""
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtype != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtype == "O"]
    cat_cols = [col for col in cat_cols + num_but_cat if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O" and col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car

################################################
# 3. PREPROCESSING PIPELINE
################################################
def air_quality_pipeline(df, test_size=0.2, scaler_type="standard", random_state=42):
    # TIME
    if "TIME" in df.columns:
        df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
        df.sort_values("TIME", inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Grab columns
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    target_col = "Status"

    # Remove target and TIME from categorical/numerical lists
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    if target_col in num_cols:
        num_cols.remove(target_col)
    if "TIME" in num_cols:
        num_cols.remove("TIME")  # <-- burası çok önemli

    # One-hot encoding
    df = one_hot_encoder(df, cat_cols)

    # Outlier handling
    for col in num_cols:
        replace_with_thresholds(df, col)

    # Scaling
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Split X and y
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler


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


