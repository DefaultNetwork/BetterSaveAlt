import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Function to convert time series to supervised learning format
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# -------------------------
# Data Loading and Configuration
# -------------------------

# Define the path to the data directory relative to the script's location
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data')

# Load the cleaned dataset
df = pd.read_csv(os.path.join(data_dir, 'Cleaned_Energy_Data.csv'))

# Use the target variable: energy_consumption
target = 'Residual load [MWh] Calculated resolutions'
data = df[[target]].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define the look-back window and create the dataset
look_back = 3
X, Y = create_dataset(scaled_data, look_back)
# Reshape X to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data into training and testing sets (80/20 split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(X_train, Y_train, epochs=20, batch_size=1, verbose=2)

# Evaluate the model on both training and testing sets
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_rmse = np.sqrt(mean_squared_error(Y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(Y_test, test_predict))
print("LSTM Model Evaluation:")
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Inverse transform predictions and actual test values for plotting
test_predict_inv = scaler.inverse_transform(test_predict)
Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Plot actual vs. predicted values for the test set
plt.figure(figsize=(10, 5))
plt.plot(Y_test_inv, label='Actual')
plt.plot(test_predict_inv, label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('Energy Consumption')
plt.title('LSTM Predictions vs Actual')
plt.legend()
plt.show()
