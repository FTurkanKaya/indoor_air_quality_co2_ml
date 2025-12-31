import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def plot_predictions(y_true, y_pred, title="Predicted vs Actual CO2"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual CO2")
    plt.ylabel("Predicted CO2")
    plt.title(title)
    plt.show()

def feature_importance(model, feature_names, top_n=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
        df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)
        plt.figure(figsize=(8,6))
        plt.barh(df_imp["feature"], df_imp["importance"])
        plt.gca().invert_yaxis()
        plt.title("Top Feature Importances")
        plt.show()
    else:
        print("Model does not have feature_importances_ attribute.")
