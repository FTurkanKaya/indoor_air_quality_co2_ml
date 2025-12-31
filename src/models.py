from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Regression models
def build_regression_models():
    return {
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=200),
        "XGBoost": XGBRegressor(eval_metric='rmse'),
        "LightGBM": LGBMRegressor()
    }

def regression_hyperparameter_optimization(X_train, y_train, X_test, y_test, regressors, scoring="r2"):
    best_models = {}
    scores = {}
    for name, model in regressors.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        best_models[name] = model
        scores[name] = score
        print(f"{name} {scoring} score: {score:.4f}")
    return best_models, scores

# Scaling & train/test split
def preprocess_for_model(df, target_col="CO2", scaler_type="standard", test_size=0.2, random_state=42):
    df = df.copy()
    X = df.drop([target_col, "TIME"], axis=1)
    y = df[target_col]

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")

    num_cols = X.select_dtypes(include=["float64","int64"]).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, scaler
