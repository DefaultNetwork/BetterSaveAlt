import pandas as pd
import os

# Load the preprocessed energy consumption and generation datasets
# Define the path to the data directory relative to the script's location
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data')

# Load the preprocessed energy consumption and generation datasets
consumption_df = pd.read_csv(os.path.join(data_dir, 'energy_consumption_preprocessed.csv'))
generation_df = pd.read_csv(os.path.join(data_dir, 'energy_generation_preprocessed.csv'))

# Check if 'Date' column exists in both DataFrames
print("Consumption DataFrame columns:", consumption_df.columns)
print("Generation DataFrame columns:", generation_df.columns)

# Merge the datasets on the 'Date' column if it exists in both DataFrames
if 'Date' in consumption_df.columns and 'Date' in generation_df.columns:
    merged_df = pd.merge(consumption_df, generation_df, on='Date', how='inner')
    # Display the first few rows of the merged dataset to verify
    print(merged_df.head())
    # Save the cleaned and merged dataset for further processing
    merged_df.to_csv('data/Cleaned_Energy_Data.csv', index=False)
else:
    print("Error: 'Date' column not found in one of the DataFrames.")
