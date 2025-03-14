import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Assume 'df' has been loaded and processed as in previous steps
df = pd.read_csv('data/Cleaned_Energy_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Define target and exogenous features
target_column = 'energy_consumption'
exogenous_features = ['energy_generation']  # Modify if needed

y = df[target_column]
X_exog = df[exogenous_features]

# Define model parameters (adjust as necessary)
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

# Split the data into training and test sets (80/20 split)
train_size = int(len(y) * 0.8)
train_y = y.iloc[:train_size]
test_y = y.iloc[train_size:]
train_exog = X_exog.iloc[:train_size]
test_exog = X_exog.iloc[train_size:]

# Fit the SARIMAX model on the training set
sarimax_model_train = sm.tsa.statespace.SARIMAX(train_y, exog=train_exog, order=order, seasonal_order=seasonal_order)
sarimax_model_train_fit = sarimax_model_train.fit(disp=False)

# Forecast for the test period
forecast = sarimax_model_train_fit.predict(start=train_size, end=len(y)-1, exog=test_exog)

# Evaluate the performance using MSE and RMSE
mse = mean_squared_error(test_y, forecast)
rmse = np.sqrt(mse)

print("SARIMAX Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
