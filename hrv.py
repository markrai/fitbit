import pandas as pd
import matplotlib.pyplot as plt

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

# Plot the average RMSSD for each month
plt.figure(figsize=(14, 7))
plt.plot(monthly_avg['month_year'], monthly_avg['rmssd'], marker='o')
plt.title('Average HRV (RMSSD) per Month')
plt.xlabel('Month/Year')
plt.ylabel('Average RMSSD')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the results to CSV files for future reference
hourly_avg.to_csv('average_stress_per_hour.csv', index=False)
monthly_avg.to_csv('average_hrv_per_month.csv', index=False)
yearly_avg.to_csv('average_hrv_per_year.csv', index=False)
weekly_avg.to_csv('average_stress_per_day_of_week.csv', index=False)
