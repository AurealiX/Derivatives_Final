import yfinance as yf
import pandas as pd

# Define the ticker symbol and the date range

# Parameters
ticker = 'SPY'
start_date = '2021-01-01'
end_date = '2023-12-31'
atr_window = 10
window_multiplier = 5

# Download historical data for SPY
download_start = pd.to_datetime(start_date) - pd.Timedelta(days=atr_window)
data = yf.download(ticker, start=download_start, end=end_date)

# Calculate the True Range (TR)
data['High-Low'] = data['High'] - data['Low']
data['High-Close'] = abs(data['High'] - data['Close'].shift())
data['Low-Close'] = abs(data['Low'] - data['Close'].shift())
data['TR'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)

# Calculate the ATR
data['ATR'] = data['TR'].rolling(window=atr_window).mean()

# Calculate the lower bound for the stock every Friday
data['Lower_Bound'] = data['Close'] - window_multiplier * data['ATR']

# Filter to get only Fridays
friday_data = data[data.index.weekday == 4]
friday_data.reset_index(inplace=True)
friday_data = friday_data[friday_data["Date"] >= start_date]
friday_data = friday_data[['Date', 'ATR', 'Lower_Bound']]

# Export lower bound data to a CSV file
friday_data.to_csv('lower_bound.csv')

print(f'Exported ATR Bands lower bound data to lower_bound.csv')
print(f'''Parameters: 
      - Ticker: SPY
      - Start Date: {start_date}
      - End Date: {end_date}
      - ATR Window: {atr_window} days
      - Window Multiplier: {window_multiplier}''')