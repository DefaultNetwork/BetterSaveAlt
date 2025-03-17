from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Function to load the model (if using dynamic forecasting)
def load_model():
    with open('sarimax_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@app.route('/predict', methods=['POST'])
def predict():
    # Get the request JSON data
    data = request.get_json()

    # Parse forecast parameters from the input
    # For example, forecast_steps (number of future periods) and exogenous data if needed
    forecast_steps = data.get("forecast_steps", 5)  # default to 5 steps ahead
    exog_future = data.get("exog_future")  # should be provided as a list or similar

    # If using dynamic forecasting, load the model and compute the forecast
    model = load_model()
    # Convert exogenous future data to a DataFrame if necessary
    if exog_future:
        exog_future = pd.DataFrame(exog_future)
    else:
        exog_future = None

    # Get forecast as a dictionary
    forecast = forecast_sarimax(model, forecast_steps, exog_future)

    return jsonify({"forecast": forecast})

if __name__ == '__main__':
    app.run(debug=True)
