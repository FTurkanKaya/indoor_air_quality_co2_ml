# Indoor Air Quality CO₂ Prediction with Machine Learning

## Project Overview
This project focuses on predicting **CO₂ levels in indoor environments** using machine learning techniques.  
By leveraging environmental sensor data such as temperature, humidity, and particulate matter (PM2.5, PM10), the goal is to provide **insights into indoor air quality** and help monitor and improve environmental conditions.

---

## Problem Definition
Indoor air quality is a critical factor for health and comfort. Accurately predicting CO₂ levels allows building managers and smart home systems to optimize ventilation and maintain safe air quality.

- **Machine Learning Type:** Supervised Learning  
- **Problem Type:** Regression (continuous target: CO₂ ppm)  
- **Input Features:** Temperature, Humidity, PM2.5, PM10, Timestamp (and optionally Occupancy or Room info)  
- **Target Variable:** CO₂ concentration (ppm)

---

## Dataset
- **Name:** Indoor Air Quality Dataset (Figshare)  
- **Source:** [Figshare](https://figshare.com/articles/dataset/Indoor_Air_Quality_Dataset_PM_sub_2_5_sub_PM_sub_10_sub_CO_sub_2_sub_TEMPERATURE_HUMIDITY_/27280983?utm_source=chatgpt.com)  
- **Measurement Interval:** ~15 seconds  
- **Duration:** Approximately six days of continuous data  
- **Variables:**  
  - Timestamp  
  - Temperature (°C)  
  - Humidity (%)  
  - PM2.5, PM10  
  - CO₂ (ppm) — target
- **Exploratory analysis revealed:**
 Right-skewed PM distributions
 Some extreme outliers in CO2 and particulate matter
 Unscaled HUMIDITY values exceeding expected range
---

## Methodology
The project follows a structured ML workflow:

1. **Exploratory Data Analysis (EDA)**
   - Inspect dataset structure and statistics  
   - Visualize trends and correlations  
   - Detect anomalies or outliers

2. **Feature Engineering**
   - Extract time-based features: hour, day of week, weekday/weekend, etc.  
   - Create interaction features: temperature × humidity, PM × humidity  
   - Optional: occupancy or room information

3. **Modeling**
   - Baseline Linear Regression  
   - Random Forest Regressor  
   - Gradient Boosting Regressor (XGBoost / LightGBM)  
   - Multi-feature regression with hyperparameter tuning

4. **Evaluation**
   - RMSE, MAE, R² Score  
   - Feature importance analysis  
   - Comparison across models

---

## Project Structure
indoor-air-quality-co2-ml/
│
├── data/
│ ├── raw/
│ │ └── indoor_air_quality.csv
│ └── processed/
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_modeling.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── models.py
│ └── evaluation.py
│
├── results/
│ ├── figures/
│ └── metrics/
│
├── README.md
└── requirements.txt

---

## Pipeline

The end-to-end pipeline consists of:
 1- **Data Loading & Validation –** Load CSV, check for nulls and types.
 2- **Exploratory Data Analysis (EDA) –** Distribution plots, correlation analysis.
 3- **Data Cleaning –** Clip negatives, replace extreme outliers, interpolate where needed.
 4- **Feature Engineering**
  **- Time-based:** hour, day of week, weekend flag
  **- Interaction:** temp_humidity, pm_total
  **-** One-hot encoding for categorical variables
 5- **Train/Test Split –** StandardScaler applied to numerical features.

 6- **Model Training –** DecisionTree, RandomForest, XGBoost, LightGBM

 7- **Cross-Validation –** 5-fold CV RMSE for model selection

 8- **Hyperparameter Optimization –** GridSearchCV to tune key parameters

 9- **Evaluation –** RMSE, MAE, R² on test set

10- **Interpretation –** Feature importance visualization

---

## Models
**Model	      CV RMSE (best)	   Test RMSE	   Test R²**
**RandomForest**	7.48	             5.70	         0.986
**XGBoost**	      8.06	             6.68	         0.981
**LightGBM**	    8.63	             7.08	         0.979

**Best model:** RandomForest
RandomForest provides accurate predictions with low error and high R².

---

## Feature Importance
Top features for RandomForest:

**Feature importance bar chart**
<img width="800" height="600" alt="Top_FeatureImportance" src="https://github.com/user-attachments/assets/00f69d93-fcec-4868-9fa8-e117b1ed892f" />

**Interpretation:** Humidity and time-related features are the most influential for CO2 prediction.

---

## Visualizations

**CO2 over time**
<img width="1007" height="591" alt="CO2 Concentration Over Time" src="https://github.com/user-attachments/assets/708de868-f671-4f85-b624-4de3286f96c3" />

**Distribution histograms for numeric features**
<img width="640" height="480" alt="Distribution_CO2" src="https://github.com/user-attachments/assets/6284a3cd-fa7d-4bfd-9e68-474671e16bc2" />
<img width="640" height="480" alt="Distribution_humudity" src="https://github.com/user-attachments/assets/b1164e0e-f318-4a9c-b6c9-313ecdf7ebea" />
<img width="640" height="480" alt="Distribution_PM2_5" src="https://github.com/user-attachments/assets/cdc021ae-803f-489f-ad11-a9efb75cda8a" />
<img width="640" height="480" alt="Distribution_PM10" src="https://github.com/user-attachments/assets/0863224a-a81f-4698-8aa3-b3cf05e58010" />
<img width="640" height="480" alt="Distribution_temperature" src="https://github.com/user-attachments/assets/4ea51303-fdd7-4a28-bf4c-dd3993e094b7" />

**Predicted vs Actual CO2**
First 10 predictions vs actual:
Actual: 408.00, Predicted: 405.61
Actual: 333.00, Predicted: 330.34
Actual: 306.00, Predicted: 306.00
Actual: 402.00, Predicted: 402.47
Actual: 415.00, Predicted: 416.27
Actual: 473.00, Predicted: 475.31
Actual: 415.00, Predicted: 414.15
Actual: 498.00, Predicted: 419.29
Actual: 452.00, Predicted: 452.38

--- 

## Usage
import joblib
from evaluation import plot_predictions, feature_importance

# Load trained model
rf_model = joblib.load("models/random_forest_best.pkl")

# Predict
y_pred = rf_model.predict(X_test)

# Evaluate
metrics = regression_metrics(y_test, y_pred)
print(metrics)

# Visualize predictions
plot_predictions(y_test, y_pred, title="RandomForest Predictions")

# Feature importance
feature_importance(rf_model, X_train.columns)

---

## Technologies Used
- Python  
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn  
- XGBoost / LightGBM (optional)

---

## Key Outcomes
- Accurate prediction of indoor CO₂ levels  
- Insights on relationships between environmental factors and air quality  
- Demonstration of feature engineering and regression modeling for IoT sensor data

---

## Future Improvements
- Include occupancy data for better predictions  
- Hyperparameter optimization and model ensembling  
- Deploy as a real-time CO₂ monitoring tool  
- Extend to multi-room or multi-building datasets

---

## Notes

HUMIDITY scaling is handled only in preprocess_for_model to avoid double-scaling.

All functions are modular, stored in src/ folder for clean OOP-style import and reuse.

GridSearchCV ensures the best hyperparameters are selected using cross-validation.

---

## Author
Developed as a **portfolio-quality machine learning project** focusing on **indoor air quality monitoring and environmental prediction**.
