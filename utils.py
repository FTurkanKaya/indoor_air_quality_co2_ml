################################################
# IMPORTS
################################################
# Temel kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn ön işleme ve pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Sklearn sınıflandırma modelleri
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier

# Sklearn regresyon modelleri (gerekiyorsa)
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Boosting sınıflandırma modelleri
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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
# 3. OUTLIER FUNCTIONS
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
# 4. COLUMN GRAB FUNCTION
################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """Get categorical, numerical, and categorical-but-cardinal column names."""
    # Categorical
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]

    # Numerical but categorical
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtype != "O"]

    # Categorical but cardinal
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtype == "O"]

    # Final categorical list
    cat_cols = [col for col in cat_cols + num_but_cat if col not in cat_but_car]

    # Numerical columns
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O" and col not in num_but_cat]

    # Remove TIME if exists
    if "TIME" in num_cols:
        num_cols.remove("TIME")
    if "TIME" in cat_cols:
        cat_cols.remove("TIME")

    return cat_cols, num_cols, cat_but_car

################################################
# 5. PREPROCESSING PIPELINE
################################################
def add_time_features(df, time_col="TIME"):
    """Timestamp bazlı özellikler ekler."""
    df = df.copy()
    if time_col in df.columns:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        df["hour"] = ts.dt.hour
        df["dayofweek"] = ts.dt.dayofweek
        df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    return df


def add_interaction_features(df):
    """Çevresel parametreler arasında etkileşim özellikleri ekler."""
    df = df.copy()
    if {"TEMPERATURE", "HUMIDITY"}.issubset(df.columns):
        df["temp_humidity"] = df["TEMPERATURE"] * df["HUMIDITY"]
    if {"PM2.5", "PM10"}.issubset(df.columns):
        df["pm_total"] = df["PM2.5"] + df["PM10"]
    return df


def air_quality_pipeline(df, test_size=0.2, scaler_type="standard", random_state=42):
    df = df.copy()

    # --- TIME feature engineering ---
    if "TIME" in df.columns:
        df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
        df.sort_values("TIME", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Saat, gün, haftasonu gibi özellikler ekle
        df["hour"] = df["TIME"].dt.hour
        df["dayofweek"] = df["TIME"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        # Dayofweek için one-hot encoding
        dayofweek_dummies = pd.get_dummies(df["dayofweek"], prefix="dayofweek")
        df = pd.concat([df, dayofweek_dummies], axis=1)

    # --- Interaction features ---
    if {"TEMPERATURE", "HUMIDITY"}.issubset(df.columns):
        df["temp_humidity"] = df["TEMPERATURE"] * df["HUMIDITY"]
    if {"PM2.5", "PM10"}.issubset(df.columns):
        df["pm_total"] = df["PM2.5"] + df["PM10"]

    # --- Grab column names ---
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    target_col = "Status"

    # --- Remove target and TIME from feature lists ---
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    if target_col in num_cols:
        num_cols.remove(target_col)
    if "TIME" in num_cols:
        num_cols.remove("TIME")

    # --- One-hot encoding categorical columns ---
    df = one_hot_encoder(df, cat_cols)

    # --- Outlier handling for numerical features ---
    for col in num_cols:
        replace_with_thresholds(df, col)

    # --- Scaling ---
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")

    df[num_cols] = scaler.fit_transform(df[num_cols])

    # --- Split X and y ---
    X = df.drop([target_col, "TIME"], axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler


######################################################
# 6. Base Models
######################################################

def base_models_pipeline(X, y, preprocessor, scoring="roc_auc", cv=3):
    """
    X, y: veri seti
    preprocessor: ön işleme pipeline (senin pre_step4)
    scoring: metric, default roc_auc
    cv: cross-validation fold sayısı
    """
    print("Base Models...\n")

    # Hızlı çalışacak modeller
    classifiers = [
        ("LR", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("CART", DecisionTreeClassifier()),
        ("RF", RandomForestClassifier(n_estimators=200)),
        ("LightGBM", LGBMClassifier()),
        ("XGBoost", XGBClassifier(eval_metric='logloss'))
    ]

    results = {}

    for name, clf in classifiers:
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", clf)
        ])
        cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        mean_score = cv_results['test_score'].mean()
        print(f"{name} {scoring}: {mean_score:.4f}")
        results[name] = mean_score

    return results


# Hyperparameter Optimization
knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]), ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


################################################
# 7. REGRESSION MODELS
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
# 8. PLOTTING
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


