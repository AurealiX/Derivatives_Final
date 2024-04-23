import yfinance as yf
import pandas as pd   

def ATRB(data, mode = 'download', atr_window = 10, window_multiplier = 4.5, output_file_name = None):
      if mode == 'download':
            print("Using download mode, input data should be a list of 3 elements: ticker, start_date, end_date")
            ticker, start_date, end_date = data
            print(f'''Parameters: 
                  - Ticker: {ticker}
                  - Start Date: {start_date}
                  - End Date: {end_date}
                  - ATR Window: {atr_window} days
                  - Window Multiplier: {window_multiplier}''')
            if output_file_name:
                  file_name = output_file_name
                  print(f"ATR Bands data will be exported to {file_name}")
                  export = True

            # Download historical data for SPY
            download_start = pd.to_datetime(start_date) - pd.Timedelta(days=atr_window)
            data = yf.download(ticker, start=download_start, end=end_date)
      elif mode == 'data':
            print("Using data mode, input data should be a dataframe with columns: Date, Open, High, Low, Close")

      # Calculate the True Range (TR)
      data['High-Low'] = data['High'] - data['Low']
      data['High-Close'] = abs(data['High'] - data['Close'].shift())
      data['Low-Close'] = abs(data['Low'] - data['Close'].shift())
      data['TR'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)

      # Calculate the ATR
      data['ATR'] = data['TR'].rolling(window=atr_window).mean()

      # Calculate the lower bound for the stock every Friday
      data['Lower_Bound'] = data['Close'] - window_multiplier * data['ATR']
      data['Upper_Bound'] = data['Close'] + window_multiplier * data['ATR']

      # Filter to get only Fridays
      friday_data = data[data.index.weekday == 4]
      friday_data.reset_index(inplace=True)
      friday_data = friday_data[friday_data["Date"] >= start_date]
      friday_data = friday_data[['Date', 'ATR', 'Lower_Bound', 'Upper_Bound']]

      # Export lower bound data to a CSV file
      if export:
            friday_data.to_csv(file_name, index=False)
            print(f'Exported ATR Bands data to {file_name}')
      
      return friday_data