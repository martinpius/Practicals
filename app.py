import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import numpy as np

# Download stock price data (e.g., Apple)

data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
close_prices = data['Close']

# Log returns
log_returns = np.log(close_prices).diff().dropna()

# Plot log returns
log_returns.plot(figsize=(10, 4), title='Log Returns of AAPL Stock')
plt.grid()
plt.show()

# ACF & PACF
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
sm.graphics.tsa.plot_acf(log_returns, lags=30, ax=ax[0])
sm.graphics.tsa.plot_pacf(log_returns, lags=30, ax=ax[1])
plt.tight_layout()
plt.show()

# Fit AR(1)
model_ar = ARIMA(log_returns, order=(1, 0, 0)).fit()
print(model_ar.summary())

# Fit MA(1)
model_ma = ARIMA(log_returns, order=(0, 0, 1)).fit()
print(model_ma.summary())

# Fit ARMA(1,1)
model_arma = ARIMA(log_returns, order=(1, 0, 1)).fit()
print(model_arma.summary())

# Plot residuals
model_arma.resid.plot(title='ARMA(1,1) Residuals', figsize=(10, 4))
plt.show()
