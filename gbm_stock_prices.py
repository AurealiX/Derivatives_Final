import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


class GBMPrices:
    def __init__(self):
        pass

    def download_history(self, ticker, start, end):
        # Download historical data for SPY
        data = yf.download(ticker, start=start, end=end)
        daily_returns = data['Adj Close'].pct_change().dropna()
        mean_daily_return = np.mean(daily_returns)
        annual_return = mean_daily_return * 252
        daily_volatility = np.std(daily_returns)
        annual_volatility = daily_volatility * np.sqrt(252)
        last_price = data['Close'][-1]
        
        self.S0, self.mu, self.sigma = last_price, annual_return, annual_volatility
        self.history = data
        return data

    def simulate_one_path(self, current_end, target_end):
        # Calculate the number of trading days left from end_date to target_end_date
        sim_days = np.busday_count(current_end, target_end)

        # Calculate the number of hourly steps considering 6.5 trading hours per day
        dt = 1 / (6.5 * 2)  # One hour as a fraction of a trading day
        N = int(sim_days * (6.5 * 2))  # Total number of hourly trading intervals
        
        return self.GBM_paths(self.S0, self.mu / 252 / (6.5*2), self.sigma / np.sqrt(252 / (6.5*2)), dt, N, 1)

    def simulate_multiple_path(self, current_end, target_end, simulations):
        sim_days = np.busday_count(current_end, target_end)
        dt = 1 / (6.5 * 2)
        N = int(sim_days * (6.5 * 2))
        return self.GBM_paths(self.S0, self.mu / 252 / (6.5*2), self.sigma / np.sqrt(252 / (6.5*2)), dt, N, simulations)

    def GBM_paths(self, S, mu, sigma, dt, steps, Npaths):
        dt = 1 / 252 / 6.5  # Adjust dt for hourly intervals in trading days
        res = np.zeros((Npaths, steps + 1))
        res[:, 0] = S
        for path in range(Npaths):
            for time in range(1, steps + 1):
                S_time = res[path, time - 1]
                normal_shock = np.random.normal()
                dS = S_time * (mu * dt + sigma * normal_shock * np.sqrt(dt))
                res[path, time] = S_time + dS
        return res

    def simu_high_low_close(self, current_end, target_end):
        one_path_gbm = self.simulate_one_path(current_end, target_end)
        business_days = pd.date_range(
            start = (pd.to_datetime(current_end) + np.timedelta64(1,'D')).strftime('%Y-%m-%d'), 
            end = (pd.to_datetime(target_end) + np.timedelta64(1,'D')).strftime('%Y-%m-%d'), 
            freq = 'B').to_numpy()
        day_idx = np.repeat(business_days, int(6.5*2))
        one_path_df = pd.DataFrame(one_path_gbm.T, columns=['Prices'])[1:]
        one_path_df['Date'] = day_idx

        def helper(group):
            low, high, close = np.min(group['Prices']), np.max(group['Prices']), list(group['Prices'])[-1]
            return pd.Series({'High': high, 'Low': low, 'Close': close})
        
        return one_path_df.groupby(['Date']).apply(lambda group: helper(group)).reset_index()