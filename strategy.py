import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the trading strategy class
class MomentumStrategy:
    def __init__(self, data, short_window=20, long_window=50, risk_free_rate=0.01):
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.risk_free_rate = risk_free_rate
        self.signals = None
        self.positions = None

    # Generate trading signals
    def generate_signals(self):
        self.data['short_mavg'] = self.data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        self.data['long_mavg'] = self.data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        self.data['signals'] = 0
        self.data['signals'][self.short_window:] = np.where(self.data['short_mavg'][self.short_window:] > self.data['long_mavg'][self.short_window:], 1, 0)
        self.signals = self.data['signals']

    # Risk management and position sizing
    def position_sizing(self, capital, risk_per_trade):
        return capital * risk_per_trade

    # Backtesting the strategy
    def backtest(self, initial_capital=10000, risk_per_trade=0.01):
        self.generate_signals()
        self.data['positions'] = self.data['signals'].diff()
        self.data['strategy_returns'] = self.data['Close'].pct_change() * self.data['signals'].shift(1)
        self.data['portfolio_value'] = initial_capital * (1 + self.data['strategy_returns']).cumprod()
        self.data['drawdown'] = self.data['portfolio_value'] / self.data['portfolio_value'].cummax() - 1
        self.data['max_drawdown'] = self.data['drawdown'].min()
        self.calculate_performance_metrics(initial_capital)

    # Calculate performance metrics
    def calculate_performance_metrics(self, initial_capital):
        total_return = self.data['portfolio_value'].iloc[-1] - initial_capital
        annualized_return = (total_return / initial_capital) * (252 / len(self.data))
        annualized_volatility = self.data['strategy_returns'].std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        print("Total Return: {}".format(total_return))
        print("Annualized Return: {}".format(annualized_return))
        print("Annualized Volatility: {}".format(annualized_volatility))
        print("Sharpe Ratio: {}".format(sharpe_ratio))
        print("Max Drawdown: {}".format(self.data['max_drawdown']))

# Sample data generation for demonstration
def generate_sample_data(num_days=100):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=num_days)
    prices = np.random.normal(loc=100, scale=1, size=num_days).cumsum()
    return pd.DataFrame(data={'Close': prices}, index=dates)

# Main execution block
if __name__ == "__main__":
    sample_data = generate_sample_data()
    strategy = MomentumStrategy(data=sample_data)
    strategy.backtest()