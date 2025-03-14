import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the preprocessed energy consumption and generation datasets
# Define the path to the data directory relative to the script's location
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data')

# Load the cleaned dataset
df = pd.read_csv(os.path.join(data_dir, 'Cleaned_Energy_Data.csv'))

# Generate summary statistics
summary_stats = df.describe()

# Display the summary statistics
print(summary_stats)

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Optionally, set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Plot the trends in energy consumption and generation over time
plt.figure(figsize=(12, 6))
# Replace 'energy_consumption' and 'energy_generation' with the actual column names if different
plt.plot(df.index, df['Residual load [MWh] Calculated resolutions'], label='Residual load [MWh] Calculated resolutions')
plt.plot(df.index, df['Photovoltaics [MWh] Calculated resolutions'], label='Photovoltaics [MWh] Calculated resolutions')
plt.xlabel('Year')
plt.ylabel('Energy (units)')
plt.title('Residual load and Photovoltaics Generation Over Time')
plt.legend()
plt.show()

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values in Each Column:")
print(missing_values)

# Display basic information about the dataframe
print("\nDataframe Info:")
print(df.info())

# Generate summary statistics again to inspect distributions
print("\nSummary Statistics:")
print(df.describe())

# Visualize missing data using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Create box plots for numeric columns to detect outliers
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(12, len(numeric_columns) * 3))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(len(numeric_columns), 1, i)
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot for {col}')
plt.tight_layout()
plt.show()
