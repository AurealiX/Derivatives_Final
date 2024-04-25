import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import brentq

# Load historical data
data = pd.read_csv('daily-treasury-rates.csv', parse_dates=['Date'], index_col='Date')
data = data['4 WEEKS COUPON EQUIVALENT']
data = data.sort_index()

weekly_data = data.resample('W-FRI').last()
weekly_data = weekly_data.sort_index()


class AmericanPutOptions:
    def __init__(self):
        # Parameters
        self.T = 1/52  # Maturity in years for each weekly option (one week)
        self.N = 7  # Number of steps in the binomial model for each week

    def load_data(self, filename):
        data = pd.read_csv(filename, parse_dates=['Date'])[['Date', 'Price', 'K']]
        data.columns = ['Date', 'Market_Put', 'Strike_Price']
        self.data = data
        return data
    
    def binomial_american_put_dynamic(self, S_0, K, T, r, N, sigma):
        dt = T / N
        r_per_period = np.exp(r * dt) - 1
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        assert 0 <= p <= 1, "Probability p is out of bounds"
        
        stock_tree = np.zeros((N+1, N+1))
        option_tree = np.zeros((N+1, N+1))

        stock_tree[0, 0] = S_0 
        for i in range(1, N+1):
            for j in range(i+1):
                if j == i:
                    stock_tree[j, i] = stock_tree[j-1, i-1] * d
                else:
                    stock_tree[j, i] = stock_tree[j, i-1] * u
        option_tree[:, N] = np.maximum(K - stock_tree[:, N], 0)

        for i in range(N-1, -1, -1):
            for j in range(i+1):
                option_tree[j, i] = (p * option_tree[j, i+1] + (1 - p) * option_tree[j+1, i+1]) / (1 + r_per_period)
                # American option feature - check for early exercise
                option_tree[j, i] = np.maximum(option_tree[j, i], K - stock_tree[j, i])
        return option_tree[0, 0]

    def get_option_prices(self, data, a, sigma):
        output = data[['Close', 'Lower_Bound', 'IR']].copy()
        output['Strike_Price'] = output.apply(lambda row: round(min(row['Lower_Bound'], row['Close'] * a)), axis=1)

        output['Put_Price'] = output.apply(
            lambda row: self.binomial_american_put_dynamic(
                row['Close'], row['Strike_Price'], self.T, row['IR'], self.N, sigma), axis=1)

        # keep only fridays
        output = output[output.index.weekday == 4]
        return output
    
    def get_implied_sigma(self, data, a):
        temp = data[['Close', 'Lower_Bound', 'IR']].copy()
        temp['Strike_Price'] = temp.apply(lambda row: round(min(row['Lower_Bound'], row['Close'] * a)), axis=1)
        temp = temp.reset_index(names=['Date'])

        simu_vs_mrk = pd.merge(temp, self.data, on=['Date', 'Strike_Price'], how="inner")
        # helper function
        def find_implied_vol(S0, K, T, r, market_price, n):
            objective = lambda vol: self.binomial_american_put_dynamic(S0, K, T, r, n, vol) - market_price
            implied_vol = brentq(objective, 0.01, 2.0)  # Search between 1% and 200% volatility
            return implied_vol
        implied_sigma = simu_vs_mrk.apply(
            lambda row: find_implied_vol(
                row['Close'], row['Strike_Price'], self.T, row['IR'], row['Market_Put'], self.N), axis=1)
        self.implied_sigma = np.average(implied_sigma)
        return self.implied_sigma


class HLInterestRates:
    def __init__(self):
        pass

    def load_historical(self, filename):
        data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
        data = data['4 WEEKS COUPON EQUIVALENT']
        data = data.sort_index()
        data = data / 100
        weekly_data = data.resample('W-FRI').last()
        weekly_data = weekly_data.sort_index()
        self.data, self.weekly_data = data, weekly_data
        return data, weekly_data

    def ho_lee_params(self, data):
        delta_r = data.diff().dropna()
        theta = delta_r.mean() / (1/52)  # Assuming weekly data
        sigma = delta_r.std() / np.sqrt(1/52)
        self.theta, self.sigma = theta, sigma
        return theta, sigma
    
    def simulate_ho_lee(self, r0, theta, sigma, start_date, end_date, dt):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        dates = pd.date_range(start=start_date, end=end_date, freq='D')  
        rates = [r0]
        for _ in range(1, len(dates)):
            dr = theta * dt + sigma * np.sqrt(dt) * np.random.normal()
            rates.append(rates[-1] + dr)
        return pd.Series(rates, index=dates)