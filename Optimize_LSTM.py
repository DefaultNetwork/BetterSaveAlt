import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# 'look_back', 'X_train', 'Y_train', 'X_test', and 'Y_test' are assumed to be defined as in the previous LSTM step

def build_model(hp):
    model = Sequential()
    # Tune the number of LSTM units
    lstm_units = hp.Int('lstm_units', min_value=20, max_value=100, step=10)
    model.add(LSTM(lstm_units, input_shape=(look_back, 1)))

    # Optionally, add a dropout layer and tune the dropout rate
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Final output layer
    model.add(Dense(1))

    # Tune the learning rate for the Adam optimizer
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
    return model

# Set up the Keras Tuner with a random search strategy
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,                # Number of different hyperparameter combinations to try
    executions_per_trial=2,      # Run each model configuration twice for stability
    directory='lstm_tuner',      # Directory to save tuning results
    project_name='energy_forecast'
)

# Start the hyperparameter search
tuner.search(X_train, Y_train, epochs=20, batch_size=1, validation_data=(X_test, Y_test))

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters:")
print(f"LSTM units: {best_hp.get('lstm_units')}")
print(f"Dropout rate: {best_hp.get('dropout_rate')}")
print(f"Learning rate: {best_hp.get('learning_rate')}")

# Optionally, evaluate the best model on the test set
test_predictions = best_model.predict(X_test)
test_loss = tf.keras.losses.mean_squared_error(Y_test, test_predictions)
print("Test Loss (MSE) on Best Model:", np.mean(test_loss))
