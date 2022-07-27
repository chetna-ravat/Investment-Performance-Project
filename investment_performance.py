import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web

with open('portfolio.txt', 'r') as f:
    PORT_TICKERS = [ticker.strip() for ticker in f]

INDEX = ["VOO"]

portfolio = pd.DataFrame()
for ticker in PORT_TICKERS:
	portfolio[ticker] = web.DataReader(ticker, data_source="yahoo", start="2016-1-1")["Adj Close"]	

benchmark = pd.DataFrame()
for ticker in INDEX:
	benchmark[ticker] = web.DataReader(ticker, data_source="yahoo", start="2016-1-1")["Adj Close"]	


# DAILY RETURN 
port_daily_return = ((portfolio/portfolio.shift(1)) - 1) * 100
benchmark_daily_return = ((benchmark/benchmark.shift(1)) - 1) * 100

print()
print("------------DAILY RETURN----------------")
print("Portfolio daily return:")
print(port_daily_return)
print("Benchmark daily return:")
print(benchmark_daily_return)

# CUMULATIVE RETURN
port_cum_return = ((1 + port_daily_return/100).cumprod() -1) * 100
benchmark_cum_return = ((1 + benchmark_daily_return/100).cumprod() -1) * 100

print()
print("------------CUMULATIVE RETURN----------------")
print("portfolio cumulative return:")
print(port_cum_return)
print("Benchmark cumulative return:")
print(benchmark_cum_return)

# ANNUALIZED RETURN
num_trading_days = 252
port_annualized_return = (pow((1 + port_cum_return.iloc[-1]/100), (252/len(portfolio))) - 1) * 100
benchmark_annualized_return = (pow((1 + port_cum_return.iloc[-1]/100), (252/len(portfolio))) - 1) * 100

print()
print("------------ANNUALIZED RETURN----------------")
print("Portfolio annulized return: \n", port_annualized_return)
print("Benchmark annulized return: \n", benchmark_annualized_return)

# PORTFOLIO RETURN

num_stocks = len(PORT_TICKERS)
weights = np.array([100/num_stocks] * num_stocks)
benchmark_weights = np.array([100/len(INDEX)] * len(INDEX))

port_return = np.dot(weights, port_annualized_return/100)
benchmark_return = np.dot(weights, port_annualized_return/100)
print()
print("------------PORTFOLIO RETURN----------------")
print(f"Portfolio return: {port_return}")
print(f"Benchmark return: {benchmark_return}")


# COVARIANCE
covar_port = (port_daily_return/100).cov() * 252
covar_benchmark = (benchmark_daily_return/100).cov() * 252
print()
print("------------COVARIANCE----------------")
print(f"Covariance: \n {covar_port}")
print(f"Covariance: \n {covar_benchmark}")

# variance 
var_port = np.dot(weights.T, np.dot(covar_port, weights))
var_benchmark = np.dot(benchmark_weights.T, np.dot(covar_benchmark, benchmark_weights))
print()
print("------------VARIANCE----------------")
print(f"variance: \n {var_port}")
print(f"variance: \n {var_benchmark}")

# STANDARD DEVIATION
port_sd = var_port ** 0.5
benchmark_sd = var_benchmark ** 0.5
print()
print("------------STANDARD DEVIATION----------------")
print(f"Portfolio Standard Deviation: {port_sd}")
print(f"Benchmark Standard Deviation: {benchmark_sd}")

# SHARPE RATIO
rf = 0.03

port_sharpe_ratio = (port_return - rf)/port_sd
benchmark_sharpe_ratio = (benchmark_return - rf)/benchmark_sd
print()
print("------------SHARPE RATIO----------------")
print(f"Sharpe ratio of portfolio: {port_sharpe_ratio}")
print(f"Sharpe ratio of benchmark: {benchmark_sharpe_ratio}")


# PLOT
comparable_df = pd.DataFrame(portfolio)
comparable_df["Benchmark"] = benchmark["VOO"]
(comparable_df/comparable_df.iloc[0]).plot(figsize=[15,6])
plt.show()