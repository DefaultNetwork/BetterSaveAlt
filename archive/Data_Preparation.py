from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import statsmodels.api as sm
import numpy as np
import os
import tensorflow as tf



# Load the preprocessed energy consumption and generation datasets
# Define the path to the data directory relative to the script's location
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data')

# Load the cleaned dataset
df = pd.read_csv(os.path.join(data_dir, 'Cleaned_Energy_Data.csv'))

# Define the target variable (e.g., 'energy_consumption')
target_column = 'Residual load [MWh] Calculated resolutions'

# Optionally, include exogenous features for SARIMAX (e.g., 'energy_generation')
exogenous_features = ['Photovoltaics [MWh] Calculated resolutions']  # Modify as necessary

# For SARIMAX:
# Feature matrix for exogenous variables (if you choose to use them)
X_exog = df[exogenous_features]
# Target variable series
y = df[target_column]

# For LSTM:
# Create lagged features for the target variable using a look-back window
look_back = 3  # Number of past time steps to use as features

# Create a DataFrame with lagged features
lagged_data = pd.concat([df[target_column].shift(i) for i in range(1, look_back + 1)], axis=1)
lagged_data.columns = [f'lag_{i}' for i in range(1, look_back + 1)]
# Drop rows with NaN values created by shifting
lagged_data = lagged_data.dropna()
# Adjust target variable to align with lagged data
y_lagged = df[target_column].iloc[look_back:]

# For demonstration, we'll use the lagged features as the LSTM input features
X_lstm = lagged_data

# Display the prepared data samples
print("SARIMAX - Exogenous Features (first few rows):")
print(X_exog.head())

print("\nSARIMAX - Target Variable (first few rows):")
print(y.head())

print("\nLSTM - Lagged Features (first few rows):")
print(X_lstm.head())

print("\nLSTM - Aligned Target Variable (first few rows):")
print(y_lagged.head())


# Define SARIMAX model parameters
# Adjust these parameters based on your data and model diagnostics
order = (1, 1, 1)              # p, d, q parameters for ARIMA
seasonal_order = (1, 1, 1, 12)   # P, D, Q, s for seasonal component (assuming monthly seasonality)

# Build the SARIMAX model
# If you prefer not to use exogenous variables, set exog=None
sarimax_model = sm.tsa.statespace.SARIMAX(y, exog=X_exog, order=order, seasonal_order=seasonal_order)

# Fit the model
sarimax_model_fit = sarimax_model.fit(disp=False)

# Print the model summary to inspect results
print(sarimax_model_fit.summary())

# Load the preprocessed energy consumption and generation datasets
# Define the path to the data directory relative to the script's location
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data')

# Load the cleaned dataset
df = pd.read_csv(os.path.join(data_dir, 'Cleaned_Energy_Data.csv'))

# Define the target variable and look-back window
target = 'Residual load [MWh] Calculated resolutions'
look_back = 3

# Function to convert a time series to a supervised learning dataset
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

# Normalize the target variable using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[[target]])

# Create the supervised dataset
X, Y = create_dataset(scaled_data, look_back)
# Reshape X to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split the dataset into training and test sets (80/20 split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Build a simple LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, Y_train, epochs=20, batch_size=1, verbose=2)

# Evaluate the model
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
trainScore = np.sqrt(mean_squared_error(Y_train, trainPredict))
testScore = np.sqrt(mean_squared_error(Y_test, testPredict))

print("LSTM Model Evaluation:")
print(f"Train Score: {trainScore:.4f} RMSE")
print(f"Test Score: {testScore:.4f} RMSE")
