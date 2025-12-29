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

## Author
Developed as a **portfolio-quality machine learning project** focusing on **indoor air quality monitoring and environmental prediction**.
