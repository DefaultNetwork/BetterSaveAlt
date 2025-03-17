import pickle
import pandas as pd

def load_model():
    with open('sarimax_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def forecast_sarimax(model, forecast_steps, exog_future):
    # Compute forecast using the loaded model
    forecast = model.get_forecast(steps=forecast_steps, exog=exog_future)
    # Convert forecast to a DataFrame or dictionary
    forecast_df = forecast.summary_frame()
    return forecast_df.to_dict(orient='records')
