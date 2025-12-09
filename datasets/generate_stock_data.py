import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate 252 trading days (1 year)
end_date = datetime(2024, 12, 31)
start_date = end_date - timedelta(days=365)
dates = pd.date_range(start=start_date, end=end_date, freq='B')[:252]

# Initial prices
initial_prices = {
    'AAPL': 150.0,
    'MSFT': 300.0,
    'GOOGL': 120.0,
    'AMZN': 140.0,
    'SPY': 400.0
}

# Generate realistic price movements
data = {'Date': dates}

for ticker, initial_price in initial_prices.items():
    # Parameters for geometric Brownian motion
    mu = 0.0005  # drift (slight upward trend)
    sigma = 0.015  # volatility

    # Generate returns
    returns = np.random.normal(mu, sigma, len(dates))

    # Calculate prices
    prices = [initial_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    data[ticker] = np.round(prices, 2)

# Create DataFrame
df = pd.DataFrame(data)

# Add some volume data (in millions)
for ticker in initial_prices.keys():
    if ticker == 'SPY':
        volume = np.random.randint(50, 150, size=len(dates))
    else:
        volume = np.random.randint(20, 80, size=len(dates))
    df[f'{ticker}_Volume'] = volume

# Save to CSV
output_path = 'D:/Joerg/Research/slides/DataScience_3/datasets/stock_prices.csv'
df.to_csv(output_path, index=False)

print(f'Generated stock price data: {len(df)} rows')
print(f'Saved to: {output_path}')
print('\nFirst 5 rows:')
print(df.head())
print('\nLast 5 rows:')
print(df.tail())
print('\nData summary:')
print(df.describe())
