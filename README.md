# Air Quality Index Prediction README

## Overview

This project focuses on predicting the Air Quality Index (AQI) using various regression models. The AQI is calculated based on pollutant concentrations, and the models used for prediction include Decision Tree Regressor, Random Forest Regressor, Support Vector Regressor, and Extra Trees Regressor. The dataset is sourced from the Central Pollution Control Board (CPCB).

## Dataset

The dataset contains pollutant concentrations for various air quality parameters including PM2.5, PM10, NO2, O3, CO, and SO2. These concentrations are used to calculate the AQI.

### Source
- Central Pollution Control Board (CPCB)

## Dependencies

Ensure you have the following dependencies installed:

```sh
pip install numpy pandas scikit-learn matplotlib
```

## Project Steps

### 1. Data Preparation

- Load the dataset and fill missing values with zero.
- Define breakpoints for each pollutant to calculate the sub-indices and the overall AQI.

### 2. AQI Calculation

Calculate the AQI based on the pollutant concentrations using predefined breakpoints.

### 3. Data Splitting

Split the data into training and testing sets.

### 4. One-Hot Encoding

Apply One-Hot Encoding to the categorical features.

### 5. Model Training and Evaluation

Train and evaluate the following models:
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Support Vector Regressor**
- **Extra Trees Regressor**

### 6. Performance Metrics

Evaluate model performance using RMSE and R² scores.

### 7. Visualization

Visualize the actual vs predicted AQI and compare model performances.

## Code

### Importing Libraries

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
```

### Loading and Preparing Data

```python
df = pd.read_csv("Testing data.csv")
df = df.fillna(0)
df.to_csv('edited_data.csv', index=False)

co_breakpoints = [
    (0, 1, 0, 50), (1.01, 2, 51, 100), (2.01, 10, 101, 200),
    (10.01, 17, 201, 300), (17.01, 34, 301, 400), (34.01, 999999, 401, 500)
]
# (similarly for no2_breakpoints, o3_breakpoints, so2_breakpoints, pm25_breakpoints, pm10_breakpoints)
```

### AQI Calculation Functions

```python
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
    row['co_sub_index'] = calculate_sub_index(row['co'] * 1 / 1000, co_breakpoints)
    row['so2_sub_index'] = calculate_sub_index(row['so2'], so2_breakpoints)
    aqi = max(row['pm2_5_sub_index'], row['pm10_sub_index'], row['no2_sub_index'],
              row['o3_sub_index'], row['co_sub_index'], row['so2_sub_index'])
    return round(aqi)

df['AQI'] = df.apply(calculate_aqi, axis=1)
```

### Splitting and Encoding Data

```python
x1 = df.iloc[:, 7:14].values
y1 = df.iloc[:, 14:15].values

ohe = OneHotEncoder()
x_new1 = pd.DataFrame(ohe.fit_transform(x1[:, [0]]).toarray())
feature_set = pd.concat([x_new1, pd.DataFrame(x1[:, 1:14])], axis=1, sort=False)

x_train, x_test, y_train, y_test = train_test_split(feature_set, y1, test_size=0.25, random_state=100)
```

### Model Training and Evaluation

```python
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
r2_rt = metrics.r2_score(y_test, rt_y_predict)

# Support Vector Regressor
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train_svr = sc_x.fit_transform(x_train)
y_train_svr = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_train_svr, y_train_svr)
svr_y_predict = sc_y.inverse_transform(svr_reg.predict(sc_x.transform(x_test)).reshape(1, -1))
rmse_svr = sqrt(mean_squared_error(y_test, svr_y_predict.T))
r2_svr = metrics.r2_score(y_test, svr_y_predict.T)

# Extra Trees Regressor
et_regressor = ExtraTreesRegressor(n_estimators=100, random_state=15)
et_regressor.fit(x_train, y_train)
y_pred_et = et_regressor.predict(x_test)
rmse_et = sqrt(mean_squared_error(y_test, y_pred_et))
r2_et = r2_score(y_test, y_pred_et)
```

### Performance Metrics

```python
print("Error Metrics for Test data (DT):\n rmse_dt: {rmse_dt} \n r2_dt: {r2_dt}")
print("Error Metrics for Test data:\n rmse_rt: {rmse_rt} \n r2_rt: {r2_rt}")
print("Error Metrics for Test data:\n rmse_svr: {rmse_svr} \n r2_svr: {r2_svr}")
print("Error Metrics for Test data:\n rmse_svr: {rmse_et} \n r2_svr: {r2_et}")

print("evaluating on testing data:")
print("----------------------------------------")
print("models\tR^2\tRMSE")
print("DTR\t{0:.4f}\t{1:.4f}".format(r2_dt, rmse_dt))
print("RFR\t{0:.4f}\t{1:.4f}".format(r2_rt, rmse_rt))
print("SVR\t{0:.4f}\t{1:.4f}".format(r2_svr, rmse_svr))
print("ET\t{0:.4f}\t{1:.4f}".format(r2_et, rmse_et))
```

### Visualization

```python
plt.figure(figsize=(6, 4))
plt.scatter(range(len(y_test)), y_test, c='black', marker='+', label='Actual AQI')
plt.scatter(range(len(dt_y_predict)), dt_y_predict, c='pink', marker='x', label='Predicted AQI')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Decision Tree Prediction Results in Testing')
plt.legend()
plt.show()

# (Repeat similar code for Random Forest Regressor, Support Vector Regressor, Extra Trees Regressor)

models = ['DTR', 'RFR', 'SVR', 'ET']
r2_values = [0.9981, 0.9783, 0.9865, 0.9993]
colors = ['pink', 'pink', 'pink', 'pink', 'pink']

plt.figure(figsize=(6, 4))
plt.bar(models, r2_values, color=colors, width=0.2)
plt.xlabel('Models')
plt.ylabel('R^2 Value')
plt.title('Comparison of R^2 Values for Different Models in Testing Data')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

rmse_values = [4.7549, 16.0834, 12.6770, 2.9420]

plt.figure(figsize=(6, 4))
plt.bar(models, rmse_values, color=colors, width=0.2)
plt.xlabel('Models')
plt.ylabel('RMSE Value')
plt.title('Comparison of RMSE Values for Different Models in Testing Data')
plt.ylim(0, 15)
plt.grid(axis='y',
