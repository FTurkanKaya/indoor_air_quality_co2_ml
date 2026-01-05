from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

################################################
# 1. Build Regression Models
################################################
def build_regression_models():
    """
    Returns a dictionary of regression models.
    """
    return {
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=200),
        "XGBoost": XGBRegressor(eval_metric='rmse'),
        "LightGBM": LGBMRegressor()
    }


################################################
# 2. Train Models
################################################
def train_models(models_dict, X_train, y_train):
    """
    Train each model in the dictionary and return trained models.

    Parameters
    ----------
    models_dict : dict
        Dictionary of model instances
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Target variable

    Returns
    -------
    trained_models : dict
        Dictionary of trained models
    """
    trained_models = {}
    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


################################################
# 3. Hyperparameter Optimization
################################################
def regression_hyperparameter_optimization(X_train, y_train, X_test, y_test, regressors, scoring="r2"):
    """
    Fit each regressor, calculate score on test set, and return best models and scores.
    """
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
# 4. Preprocessing & Train/Test Split
################################################
def preprocess_for_model(df, target_col="CO2", scaler_type="standard", test_size=0.2, random_state=42):
    """
    Scale numerical features and split data into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    target_col : str
        Name of target column
    scaler_type : str
        'standard' for StandardScaler, 'minmax' for MinMaxScaler
    test_size : float
        Test set proportion
    random_state : int
        Random seed

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    df = df.copy()
    X = df.drop([target_col, "TIME"], axis=1)
    y = df[target_col]

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")

    num_cols = X.select_dtypes(include=["float64", "int64"]).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, scaler

################################################
# 5. Cross-Validation
################################################

def cross_validate_models(models, X_train, y_train, cv=5):
    """
    Perform cross-validation for given models on training data.

    Parameters
    ----------
    models : dict
        Dictionary of model name and model object
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    cv : int
        Number of folds for cross-validation

    Returns
    -------
    results : dict
        Cross-validation RMSE results for each model
    """
    results = {}

    for name, model in models.items():
        # Negative RMSE is returned by sklearn (convention)
        neg_rmse_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error"
        )

        rmse_scores = -neg_rmse_scores

        results[name] = {
            "RMSE_mean": rmse_scores.mean(),
            "RMSE_std": rmse_scores.std()
        }

        print(f"{name} CV RMSE: {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}")

    return results
