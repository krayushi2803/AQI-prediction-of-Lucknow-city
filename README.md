# Air Quality Index (AQI) Prediction README

## Overview

This project involves predicting the Air Quality Index (AQI) using various machine learning models. The dataset contains several pollutants' concentration levels, which are used to calculate the AQI. We implemented and compared the performance of different regression models, including Decision Tree Regressor, Random Forest Regressor, Support Vector Regressor, and Extra Trees Regressor.

## Dataset

The dataset used for this project is sourced from the Central Pollution Control Board (CPCB).

### Source
- [CPCB](https://cpcb.nic.in/)

## Dependencies

Ensure you have the following dependencies installed:

```sh
pip install numpy pandas scikit-learn matplotlib
```

## Data Preprocessing

### Handling Missing Values
The dataset had missing values, which were filled with zeros for simplicity.

### One-Hot Encoding
One-Hot Encoding was used for categorical variables to convert them into numerical values suitable for machine learning models.

## Breakpoints for AQI Calculation

The AQI is calculated based on concentration breakpoints for various pollutants as defined by regulatory authorities. The breakpoints for different pollutants are:

- **CO**: (0-50), (51-100), (101-200), (201-300), (301-400), (401-500)
- **NO2**: (0-50), (51-100), (101-200), (201-300), (301-400), (401-500)
- **O3**: (0-50), (51-100), (101-200), (201-300), (301-400), (401-500)
- **SO2**: (0-50), (51-100), (101-200), (201-300), (301-400), (401-500)
- **PM2.5**: (0-50), (51-100), (101-200), (201-300), (301-400), (401-500)
- **PM10**: (0-50), (51-100), (101-200), (201-300), (301-400), (401-500)

## Models Implemented

### 1. Decision Tree Regressor (DTR)
### 2. Random Forest Regressor (RFR)
### 3. Support Vector Regressor (SVR)
### 4. Extra Trees Regressor (ET)

## Evaluation Metrics

The models were evaluated using the following metrics:
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error

## Results

### R² Values
- **DTR**: 0.9981
- **RFR**: 0.9783
- **SVR**: 0.9865
- **ET**: 0.9993

### RMSE Values
- **DTR**: 4.7549
- **RFR**: 16.0834
- **SVR**: 12.6770
- **ET**: 2.9420

## Visualization

The results of the predictions and the comparison of R² and RMSE values for different models were visualized using matplotlib.

## Code

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("Testing data.csv")
df = df.fillna(0)

# Define breakpoints for AQI calculation
co_breakpoints = [(0, 1, 0, 50), (1.01, 2, 51, 100), (2.01, 10, 101, 200), (10.01, 17, 201, 300), (17.01, 34, 301, 400), (34.01, 999999, 401, 500)]
no2_breakpoints = [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 999999, 401, 500)]
o3_breakpoints = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 999999, 401, 500)]
so2_breakpoints = [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 999999, 401, 500)]
pm25_breakpoints = [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 999999, 401, 500)]
pm10_breakpoints = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 999999, 401, 500)]

def calculate_sub_index(concentration, breakpoints):
    for (low_conc, high_conc, low_index, high_index) in breakpoints:
        if low_conc <= concentration <= high_conc:
            return round(low_index + (high_index - low_index) / (high_conc - low_conc) * (concentration - low_conc))
    return 500

def calculate_aqi(row):
    row['pm2_5_sub_index'] = calculate_sub_index(row['pm2_5'], pm25_breakpoints)
    row['pm10_sub_index'] = calculate_sub_index(row['pm10'], pm10_breakpoints)
    row['no2_sub_index'] = calculate_sub_index(row['no2'], no2_breakpoints)
    row['o3_sub_index'] = calculate_sub_index(row['ozone'], o3_breakpoints)
    row['co_sub_index'] = calculate_sub_index(row['co']*1/1000, co_breakpoints)  # Convert from µg/m³ to mg/m³
    row['so2_sub_index'] = calculate_sub_index(row['so2'], so2_breakpoints)
    aqi = max(row['pm2_5_sub_index'], row['pm10_sub_index'], row['no2_sub_index'], row['o3_sub_index'], row['co_sub_index'], row['so2_sub_index'])
    return round(aqi)

df['pm2_5_aqi'] = df['pm2_5'].apply(lambda x: calculate_sub_index(x, pm25_breakpoints))
df['pm10_sub_aqi'] = df['pm10'].apply(lambda x: calculate_sub_index(x, pm10_breakpoints))
df['no2_sub_aqi'] = df['no2'].apply(lambda x: calculate_sub_index(x, no2_breakpoints))
df['o3_sub_aqi'] = df['ozone'].apply(lambda x: calculate_sub_index(x, o3_breakpoints))
df['co_sub_aqi'] = df['co'].apply(lambda x: calculate_sub_index(x*1/1000, co_breakpoints))
df['so2_sub_aqi'] = df['so2'].apply(lambda x: calculate_sub_index(x, so2_breakpoints))
df['AQI'] = df.apply(calculate_aqi, axis=1)

x1 = df.iloc[:, 7:14].values
y1 = df.iloc[:, 14:15].values

ohe = OneHotEncoder()
x_new1 = pd.DataFrame(ohe.fit_transform(x1[:, [0]]).toarray())
feature_set = pd.concat([x_new1, pd.DataFrame(x1[:, 1:14])], axis=1, sort=False)

x_train, x_test, y_train, y_test = train_test_split(feature_set, y1, test_size=0.25, random_state=100)

# Decision Tree Regressor
dec_tree = DecisionTreeRegressor(random_state=13)
dec_tree.fit(x_train, y_train)
dt_y_predict = dec_tree.predict(x_test)
rmse_dt = sqrt(mean_squared_error(y_test, dt_y_predict))
r2_dt = r2_score(y_test, dt_y_predict)

# Random Forest Regressor
rt_reg = RandomForestRegressor(n_estimators=1000, random_state=12)
rt_reg.fit(x_train, y_train)
rt_y_predict = rt_reg.predict(x_test)
rmse_rt = sqrt(mean_squared_error(y_test, rt_y_predict))
r2_rt = r2_score(y_test, rt_y_predict)

# Support Vector Regressor
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train_s
