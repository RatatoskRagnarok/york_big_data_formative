import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
This file finds the top 6 countries and months for cancellations

The original Jupyter notebook with inline dataframes and figures can be found at
#TODO insert link here
"""
#TODO insert jupyter notebook link

# load data with all countries
df = pd.read_csv('hotel_clean_all_countries.csv')

# Total bookings and cancellations per country + month
summary = df.groupby(['country', 'arrival_month'])['is_canceled'].agg(['count', 'sum']).reset_index()
summary.rename(columns={'count': 'total_bookings', 'sum': 'cancellations'}, inplace=True)

# Calculate cancellation rate
summary['cancellation_rate'] = summary['cancellations'] / summary['total_bookings']
sorted_countries = summary.sort_values(by=['cancellation_rate'], ascending=False)

# how many have a 100% cancellation rate?
sorted_countries = sorted_countries.drop_duplicates(subset='country')
all_cancelled = sorted_countries[sorted_countries['cancellation_rate']==1.0]
num_100 = all_cancelled.shape[0]

# results
print("Top 6 countries and months by cancellation rate")
print(sorted_countries.drop_duplicates(subset='country').head(6))
print(f"Number of countries with at least one month with a 100% cancellation rate: {num_100}")

# Try it after consolidation of smaller booking countries
df = pd.read_csv('hotel_cleaned_no_encoding.csv')

# Total bookings and cancellations per country + month
summary = df.groupby(['country', 'arrival_month'])['is_canceled'].agg(['count', 'sum']).reset_index()
summary.rename(columns={'count': 'total_bookings', 'sum': 'cancellations'}, inplace=True)

# Calculate cancellation rate
summary['cancellation_rate'] = summary['cancellations'] / summary['total_bookings']
sorted_countries = summary.sort_values(by=['cancellation_rate'], ascending=False)

# results
print("\nTop 6 countries and months by cancellation rate (after consolidation)")
print(sorted_countries.drop_duplicates(subset='country').head(6))

# sorting by number of cancellations instead
sorted_countries = summary.sort_values(by=['cancellations'], ascending=False)

print("\nTop 6 countries and months by number of cancellations")
print(sorted_countries.drop_duplicates(subset='country').head(6))

# showing the relationship between booking numbers and cancellations

c_booking_data = df.groupby('country')['is_canceled'].agg(['count', 'sum'])
c_booking_data.columns = ['total_bookings', 'cancellations']
c_booking_data = c_booking_data.sort_values(by='cancellations', ascending=False)[1:]  # remove Portugal as is outlier, easier to see pattern without

m_booking_data = df.groupby('arrival_month')['is_canceled'].agg(['count', 'sum'])
m_booking_data.columns = ['total_bookings', 'cancellations']

fig, axes = plt.subplots(2, 1, figsize=(8, 6))
axes = axes.flatten()

for i, topic in enumerate(['Country', 'Month']):
    if topic == 'Country':
        x = c_booking_data['total_bookings']
        y = c_booking_data['cancellations']
        label = 'Countries'
    else:
        x = m_booking_data['total_bookings']
        y = m_booking_data['cancellations']
        label = 'Months'

    # Find line of best fit
    a, b = np.polyfit(x, y, 1)
    line = a * x + b

    axes[i].scatter(x, y, alpha=0.6, label=label)
    axes[i].plot(x, line, color='red', linestyle='--', label='Best fit line')

    axes[i].set_title(f'Relationship between Total Bookings and Cancellations by {topic}', fontsize=14)
    axes[i].set_xlabel('Total Bookings', fontsize=12)
    axes[i].set_ylabel('Cancellations', fontsize=12)

    axes[i].legend()
    axes[i].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("total bookings vs cancellations.png")
