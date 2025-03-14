import matplotlib.pyplot as plt
import numpy as np

# Assuming the following variables have been defined in previous steps:
# - For SARIMAX:
#     test_y: Actual energy consumption values (with date index) for the test set
#     forecast: SARIMAX predictions on the test set
#     rmse: SARIMAX model RMSE (computed in Step 9)
#
# - For LSTM:
#     testPredict: LSTM predictions (on scaled data) for the test set
#     Y_test: Actual energy consumption values (on scaled data) for the test set
#     testScore: LSTM model RMSE (computed in Step 10)
#     scaler: MinMaxScaler instance used for scaling the data

# Inverse transform the LSTM predictions and actual values to original scale
lstm_predictions = scaler.inverse_transform(testPredict)
lstm_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Create a date index for LSTM predictions; assume it aligns with SARIMAX test index
# If the test set splits were done similarly, we can use test_y.index
lstm_dates = test_y.index

# Plot the actual values and model predictions for comparison
plt.figure(figsize=(14, 7))
plt.plot(test_y.index, test_y, label="Actual", color='black')
plt.plot(test_y.index, forecast, label="SARIMAX Prediction", color='blue')
plt.plot(lstm_dates, lstm_predictions, label="LSTM Prediction", color='red')
plt.title("Comparison of SARIMAX and LSTM Predictions")
plt.xlabel("Date")
plt.ylabel("Energy Consumption")
plt.legend()
plt.show()

# Print out the evaluation metrics for both models
print("SARIMAX Model RMSE:", rmse)
print("LSTM Model RMSE:", testScore)
