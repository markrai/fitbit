import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load the combined CSV file
df = pd.read_csv('data.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract hour, month, year, and day of the week from timestamp
df['hour'] = df['timestamp'].dt.hour
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Select only numeric columns for aggregation
numeric_cols = ['low_frequency', 'rmssd']

# Calculate hourly averages for stress level
hourly_avg = df.groupby('hour')['low_frequency'].mean().reset_index()

# Calculate monthly averages for HRV (rmssd)
monthly_avg = df.groupby(['year', 'month'])['rmssd'].mean().reset_index()

# Calculate yearly averages for HRV (rmssd)
yearly_avg = df.groupby('year')['rmssd'].mean().reset_index()

# Calculate average stress level for each day of the week
weekly_avg = df.groupby('day_of_week')['low_frequency'].mean().reset_index()
weekly_avg['day_name'] = weekly_avg['day_of_week'].apply(
    lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x]
)

# Create a 'month_year' column for the monthly_avg dataframe for better plotting
monthly_avg['month_year'] = monthly_avg.apply(lambda x: f"{int(x['month']):02d}/{str(int(x['year']))[2:]}", axis=1)

# Create a time index for linear regression
monthly_avg['time_index'] = np.arange(len(monthly_avg))

# Prepare data for linear regression
X = monthly_avg[['time_index']]
y = monthly_avg['rmssd']

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict RMSSD values
monthly_avg['rmssd_trend'] = model.predict(X)

# Print the average level of stress (low_frequency) for each hour
print("Average level of stress (low_frequency) over a 24-hour period for each hour:")
for index, row in hourly_avg.iterrows():
    print(f"Hour {int(row['hour']):02d}:00 - Average Low Frequency: {row['low_frequency']:.2f}")

# Print the average HRV (rmssd) per month for every year with line separators and yearly averages
print("\nAverage HRV (rmssd) per month for every year:")
current_year = None
for index, row in monthly_avg.iterrows():
    if current_year is None or current_year != row['year']:
        if current_year is not None:
            # Print yearly average for the previous year
            yearly_value = yearly_avg[yearly_avg['year'] == current_year]['rmssd'].values[0]
            print(f"\nAverage RMSSD for Year {int(current_year)}: {yearly_value:.2f}")
            print('-' * 40)
        current_year = row['year']
    print(f"Year {int(row['year'])}, Month {int(row['month']):02d} - Average RMSSD: {row['rmssd']:.2f}")

# Print yearly average for the last year in the dataset
if current_year is not None:
    yearly_value = yearly_avg[yearly_avg['year'] == current_year]['rmssd'].values[0]
    print(f"\nAverage RMSSD for Year {int(current_year)}: {yearly_value:.2f}")
    print('-' * 40)

# Print the average level of stress (low_frequency) for each day of the week
print("\nAverage level of stress (low_frequency) for each day of the week:")
print("\nHigher values denote more stress:")
for index, row in weekly_avg.iterrows():
    print(f"{row['day_name']}: Average Low Frequency: {row['low_frequency']:.2f}")

# Plot the average RMSSD for each month with trend line
plt.figure(figsize=(14, 7))
plt.plot(monthly_avg['month_year'], monthly_avg['rmssd'], marker='o', label='Average RMSSD')
plt.plot(monthly_avg['month_year'], monthly_avg['rmssd_trend'], color='red', label='Trend (Linear Regression)')
plt.title('Average HRV (RMSSD) per Month with Trend Line')
plt.xlabel('Month/Year')
plt.ylabel('Average RMSSD')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the linear regression model parameters
print(f"Linear Regression Model: RMSSD = {model.coef_[0]:.4f} * time_index + {model.intercept_:.4f}")

# Plot yearly averages of RMSSD with values on bars
plt.figure(figsize=(10, 5))
bars = plt.bar(yearly_avg['year'], yearly_avg['rmssd'], color='skyblue')

# Add text labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

plt.title('Yearly Averages of RMSSD')
plt.xlabel('Year')
plt.ylabel('Average RMSSD')
plt.xticks(yearly_avg['year'])
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Heatmap of average stress level by day of the week
plt.figure(figsize=(10, 5))
day_avg_matrix = weekly_avg.pivot_table(index='day_name', values='low_frequency')
sns.heatmap(day_avg_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Average Stress Level by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Stress Level (Low Frequency)')
plt.show()

# Heatmap of average stress level by hour of the day
plt.figure(figsize=(14, 7))
hour_avg_matrix = hourly_avg.pivot_table(index='hour', values='low_frequency')
sns.heatmap(hour_avg_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Average Stress Level by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Stress Level (Low Frequency)')
plt.show()

# Save the results to CSV files for future reference
hourly_avg.to_csv('average_stress_per_hour.csv', index=False)
monthly_avg.to_csv('average_hrv_per_month.csv', index=False)
yearly_avg.to_csv('average_hrv_per_year.csv', index=False)
weekly_avg.to_csv('average_stress_per_day_of_week.csv', index=False)

