import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the path to the CSV file in the "data" folder
csv_path = os.path.join(script_dir, "data", "Cleaned_Energy_Data.csv")

# Read the CSV file
df = pd.read_csv(csv_path)

# # Define the path to the data directory relative to the script's location
# script_dir = os.path.dirname(__file__)
# data_dir = os.path.join(script_dir, 'data')

# # Load the cleaned dataset
# df = pd.read_csv(os.path.join(data_dir, 'Cleaned_Energy_Data.csv'))

# Define the target variable (e.g., 'energy_consumption')
target = 'Residual load [MWh] Calculated resolutions'

# Optionally, include exogenous features for SARIMAX (e.g., 'energy_generation')
exog = ['Photovoltaics [MWh] Calculated resolutions']  # Modify as necessary

# For SARIMAX:
# Feature matrix for exogenous variables (if you choose to use them)
X_exog = df[exog]
# Target variable series
y = df[target]

# Split data: 80% training, 20% testing
train_size = int(len(y) * 0.8)
train_y = y.iloc[:train_size]
test_y = y.iloc[train_size:]
train_exog = X_exog.iloc[:train_size]
test_exog = X_exog.iloc[train_size:]

# Define SARIMAX model parameters (example values)
order = (1, 1, 1)                 # (p, d, q)
seasonal_order = (1, 1, 1, 12)      # (P, D, Q, s) - assuming monthly seasonality

# Build and fit the SARIMAX model on training data
model = sm.tsa.statespace.SARIMAX(train_y, exog=train_exog, order=order, seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)

# Forecast on the test set
forecast = model_fit.predict(start=train_size, end=len(y)-1, exog=test_exog)

# Evaluate the model
mse = mean_squared_error(test_y, forecast)
rmse = np.sqrt(mse)
print("SARIMAX Model Evaluation:")
print("MSE:", mse)
print("RMSE:", rmse)

# Plot actual vs. forecasted values
plt.figure(figsize=(10, 5))
plt.plot(test_y.index, test_y, label='Actual')
plt.plot(test_y.index, forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.title('SARIMAX Forecast vs Actual')
plt.legend()
plt.show()
