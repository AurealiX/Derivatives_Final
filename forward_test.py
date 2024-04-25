import numpy as np
import pandas as pd

from gbm_stock_prices import GBMPrices
from ATRB import ATRB
from holee_ir import HLInterestRates, AmericanPutOptions

class ForwardTest:
    def __init__(
            self, 
            ticker, 
            history_end_date, 
            target_end_date, 
            simu_start_date, 
            atr_window, 
            window_multiplier, 
            a,
            ir_filename, 
            mkt_filename
            ):
        self.gbm = GBMPrices()
        self.atrb = ATRB()
        self.ho_lee_ir = HLInterestRates()
        self.am_put = AmericanPutOptions()

        # Parameters
        self.ticker = ticker
        self.history_end_date, self.target_end_date = history_end_date, target_end_date
        self.history_start_date = (
            pd.to_datetime(self.history_end_date) - np.timedelta64(365, 'D')
            ).strftime('%Y-%m-%d')
        self.simu_start_date, self.simu_end_date = simu_start_date, target_end_date
        self.atr_window = atr_window
        self.window_multiplier = window_multiplier
        self.a = a

        # Download
        self.stock_data = self.gbm.download_history(
            self.ticker, 
            self.history_start_date, 
            self.history_end_date
            )
        self.ir_data, self.ir_weekly_data = self.ho_lee_ir.load_historical(ir_filename)
        self.mkt_data = self.am_put.load_data(mkt_filename)

    def simulate_stock(self):
        real_stock = self.stock_data[
            self.stock_data.index >= (
                pd.to_datetime(self.simu_start_date) - np.timedelta64(self.atr_window+10, 'D')
                ).strftime('%Y-%m-%d')]
        simu_stock = self.gbm.simu_high_low_close(self.history_end_date, self.target_end_date)
        stock_compile = pd.concat([
            simu_stock, 
            real_stock.reset_index()[simu_stock.columns]
            ]).sort_values(by=['Date']).set_index('Date')
        stock_compile = self.atrb.process(
            stock_compile, 
            self.atr_window, 
            self.window_multiplier, 
            self.simu_start_date
            )
        return stock_compile
    
    def simulate_ir(self):
        ir_theta, ir_sigma = self.ho_lee_ir.ho_lee_params(self.ir_data)
        r_0 = self.ir_data[self.ir_data.index == self.history_end_date].values[0]
        rates = self.ho_lee_ir.simulate_ho_lee(
            r_0, 
            ir_theta, 
            ir_sigma, 
            self.history_end_date, 
            self.target_end_date, 
            1 / 365)
        ir_compile = pd.concat([
            self.ir_data[self.ir_data.index < self.history_end_date], 
            rates
            ])
        return ir_compile
    
    def simulate_options(self, stock_compile, ir_compile):
        data_compile = stock_compile.copy()
        data_compile['IR'] = ir_compile
        data_compile = data_compile[['High', 'Low', 'Close', 'ATR', 'Lower_Bound', 'IR']]
        data_compile = data_compile.reindex(pd.date_range(
            start=self.simu_start_date, 
            end=self.simu_end_date, freq='D'
            )).ffill().bfill()
        implied_sigma = self.am_put.get_implied_sigma(data_compile, self.a)
        simu_option = self.am_put.get_option_prices(
            data_compile, 
            self.a, 
            implied_sigma
            ).reset_index(names=['Date'])
        option_compile = pd.merge(simu_option, self.mkt_data, how="left", on=['Date', 'Strike_Price'])
        option_compile['Put_Price'] = option_compile.apply(
            lambda row: row['Market_Put'] if not pd.isna(row['Market_Put']) else row['Put_Price'], 
            axis=1
            )
        option_compile['Next_Week_Close'] = option_compile['Close'].shift(-1)
        option_compile['Use_Threshold'] = option_compile['Lower_Bound'] < (
            option_compile['Close'] * self.a
            )
        option_compile = option_compile.drop(columns=['Market_Put']).dropna()
        return option_compile

    def forward_test(self, data):
        # Initial conditions
        initial_capital = 1000000
        account_balance = initial_capital
        contract_multiplier = 100  # Standard option contract multiplier
        margin_requirement_per_contract = 1

        # Initialize results storage
        results = []

        for _, row in data.iterrows():
            strike_price, put_price = row['Strike_Price'], row['Put_Price']
            way, next_friday_price = row['Use_Threshold'], row['Next_Week_Close']
            num_contracts = int(account_balance / (strike_price * margin_requirement_per_contract * contract_multiplier))
            premium_received = put_price * contract_multiplier * num_contracts
            account_balance += premium_received  # Add premium received
            
            if next_friday_price > strike_price:
                # Option expires worthless, keep premium
                profit_loss = 0
            else:
                # Calculate the loss based on how much in-the-money
                profit_loss = max(0, strike_price - next_friday_price) * contract_multiplier * num_contracts
                account_balance -= profit_loss  # Pay the loss
            
            results.append({
                'date': row['Date'],
                'strike_price': strike_price,
                'put_price': put_price,
                'contracts': num_contracts,
                'premium_received': premium_received,
                'profit_loss': profit_loss,
                'account_balance': account_balance,
                'use_threshold': way
            })

        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame(results)

        # Calculate performance metrics
        final_balance = results_df['account_balance'].iloc[-1]
        total_return = final_balance - initial_capital
        annual_return = (total_return / initial_capital) * (365.0 / results_df['date'].dt.dayofyear.max())  # Normalize to one year

        # Calculate weekly returns for Sharpe Ratio
        results_df['weekly_returns'] = results_df['account_balance'].pct_change()
        mean_weekly_return = results_df['weekly_returns'].mean()
        std_dev_weekly = results_df['weekly_returns'].std()

        # Sharpe Ratio, assuming risk-free rate = 0 for simplification
        sharpe_ratio = (mean_weekly_return / std_dev_weekly) * np.sqrt(52)  # 52 trading weeks in a year
        # total gain
        results_df['total_gain'] = results_df['premium_received'] - results_df['profit_loss']
        
        output_df = pd.DataFrame({
            "Final Account Balance": final_balance,
            "Total Gain": results_df['total_gain'].sum(),
            "Return on Investment (ROI)": (final_balance - initial_capital) / initial_capital,
            "Annual Return": annual_return,
            "Sharpe Ratio": sharpe_ratio,
            "Win Rate": results_df['total_gain'].apply(lambda x: 1 if x>0 else 0).sum()/results_df.shape[0],
            "Threshold used": results_df['use_threshold'].sum()/results_df.shape[0]
        }, index=[0])

        return results_df, output_df
    
    def simulation(self):
        stock_compile = self.simulate_stock()
        ir_compile = self.simulate_ir()
        return self.simulate_options(stock_compile, ir_compile)

    def simulate_returns(self, N):
        returns = pd.DataFrame()
        for _ in range(N):
            stock_compile = self.simulate_stock()
            ir_compile = self.simulate_ir()
            option_compile = self.simulate_options(stock_compile, ir_compile)
            _, output_df = self.forward_test(option_compile)
            returns = pd.concat([returns, output_df])
        return returns.reset_index(drop=True)