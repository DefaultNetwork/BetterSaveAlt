import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pmdarima as pm


# -------------------------
# Data Loading and Configuration
# -------------------------

# Define the path to the data directory relative to the script's location
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data')

# Load the cleaned dataset
df = pd.read_csv(os.path.join(data_dir, 'Cleaned_Energy_Data.csv'))

# Define target and exogenous feature names (modify these if needed)
target = 'Residual load [MWh] Calculated resolutions'
exog_feature = 'Photovoltaics [MWh] Calculated resolutions'

# -------------------------
# SARIMAX Model Section
# -------------------------

# Check for stationarity using the Augmented Dickey-Fuller test
result = adfuller(df['Residual load [MWh] Calculated resolutions'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# Check for variance stability (example using rolling standard deviation)
rolling_std = df[target].rolling(window=12).std()

# If variance is not constant, apply log transform
if rolling_std.std() > 0.1:  # Threshold can be adjusted based on specific needs
    df[target] = np.log(df[target])
    df[exog_feature] = np.log(df[exog_feature])
    print("Log transform applied due to variance instability.")
else:
    print("Variance is stable; no log transform applied.")

# This function will search for a good set of parameters.
auto_model = pm.auto_arima(df['Residual load [MWh] Calculated resolutions'], seasonal=True, m=7,
                           trace=True, error_action='ignore', suppress_warnings=True)
print(auto_model.summary())

print("---- SARIMAX Model ----")

# Prepare data for SARIMAX
X_exog = df[[exog_feature]]
y = df[target]

# Split into training and testing sets (80%/20% split)
train_size = int(len(y) * 0.8)
train_y = y.iloc[:train_size]
test_y = y.iloc[train_size:]
train_exog = X_exog.iloc[:train_size]
test_exog = X_exog.iloc[train_size:]

# Define SARIMAX model parameters (example values; adjust as needed)
order = (1, 1, 1)                # (p, d, q)
seasonal_order = (1, 1, 1, 12)     # (P, D, Q, s) assuming monthly seasonality

# Build and fit the SARIMAX model on training data
sarimax_model = sm.tsa.statespace.SARIMAX(train_y, exog=train_exog,
                                          order=order, seasonal_order=seasonal_order)
sarimax_fit = sarimax_model.fit(disp=False)

# Forecast on the test set
forecast = sarimax_fit.predict(start=train_size, end=len(y)-1, exog=test_exog)

# Evaluate the SARIMAX model
sarimax_mse = mean_squared_error(test_y, forecast)
sarimax_rmse = np.sqrt(sarimax_mse)
print("SARIMAX Evaluation:")
print(f"MSE: {sarimax_mse:.4f}")
print(f"RMSE: {sarimax_rmse:.4f}")

# Plot actual vs. forecasted values for SARIMAX
plt.figure(figsize=(10, 5))
plt.plot(test_y.index, test_y, label='Actual')
plt.plot(test_y.index, forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel(target)
plt.title('SARIMAX Forecast vs Actual')
plt.legend()
plt.show()

# -------------------------
# LSTM Model Section
# -------------------------
print("---- LSTM Model ----")

# Function to convert a time series to a supervised learning dataset
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

# Prepare data for LSTM using the target column
lstm_data = df[[target]].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(lstm_data)

# Define look-back window and create supervised dataset
look_back = 3
X_lstm, Y_lstm = create_dataset(scaled_data, look_back)
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

# Split the dataset into training and testing sets (80%/20% split)
train_size_lstm = int(len(X_lstm) * 0.8)
X_train, X_test = X_lstm[:train_size_lstm], X_lstm[train_size_lstm:]
Y_train, Y_test = Y_lstm[:train_size_lstm], Y_lstm[train_size_lstm:]

# Build a simple LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(look_back, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')

# Train the LSTM model
lstm_model.fit(X_train, Y_train, epochs=20, batch_size=1, verbose=2)

# Evaluate the LSTM model
trainPredict = lstm_model.predict(X_train)
testPredict = lstm_model.predict(X_test)
trainScore = np.sqrt(mean_squared_error(Y_train, trainPredict))
testScore = np.sqrt(mean_squared_error(Y_test, testPredict))
print("LSTM Evaluation:")
print(f"Train RMSE: {trainScore:.4f}")
print(f"Test RMSE: {testScore:.4f}")

# Inverse transform predictions for plotting
testPredict_inv = scaler.inverse_transform(testPredict)
Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Plot actual vs. predicted values for LSTM
plt.figure(figsize=(10, 5))
plt.plot(Y_test_inv, label='Actual')
plt.plot(testPredict_inv, label='Predicted')
plt.xlabel('Time Step')
plt.ylabel(target)
plt.title('LSTM Predictions vs Actual')
plt.legend()
plt.show()
