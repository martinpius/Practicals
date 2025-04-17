import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA

sns.set_theme(style="darkgrid")
np.random.seed(12938)
T = 200

# Trend-Stationary Process 
trend = 0.5 * np.arange(T)
stationary_noise = np.random.normal(0, 1, T)
trend_stationary = trend + stationary_noise

# Difference Stationary (Random Walk) 
random_walk = np.cumsum(np.random.normal(0, 1, T))

# Seasonal Stationary 
seasonal_pattern = np.sin(2 * np.pi * np.arange(T) / 12)
seasonal_noise = np.random.normal(0, 0.5, T)
seasonal_stationary = seasonal_pattern + seasonal_noise

# Plot all series 
plt.figure(figsize=(15, 8))
plt.subplot(3, 1, 1)
plt.plot(trend_stationary)
plt.title("Trend-Stationary Process")

plt.subplot(3, 1, 2)
plt.plot(random_walk)
plt.title("Difference-Stationary Process (Random Walk)")

plt.subplot(3, 1, 3)
plt.plot(seasonal_stationary)
plt.title("Seasonal Stationary Process")
plt.tight_layout()
plt.show()

#ADF TESTS 
def test_adf(series, name):
    result = adfuller(series)
    print(f"\nADF Test - {name}")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("=> Stationary" if result[1] < 0.05 else "=> Nonstationary")

test_adf(trend_stationary, "Trend-Stationary")
test_adf(random_walk, "Random Walk")
test_adf(seasonal_stationary, "Seasonal Stationary")

# Simulate ARMA(2,1)
arma_process = ArmaProcess(ar=[1, -0.5, -0.3], ma=[1, 0.7])
arma_data = arma_process.generate_sample(nsample=300)

plt.figure(figsize=(10, 4))
plt.plot(arma_data)
plt.title("Simulated ARMA(2,1) Series")
plt.show()

# ACF/PACF: Visual + Numerical 
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
plot_acf(arma_data, ax=ax[0], lags=30)
plot_pacf(arma_data, ax=ax[1], lags=30)
plt.tight_layout()
plt.show()

# Print first few ACF and PACF values numerically
acf_vals = acf(arma_data, nlags=10)
pacf_vals = pacf(arma_data, nlags=10)
print("\nACF values (first 10 lags):")
print(np.round(acf_vals, 3))
print("\nPACF values (first 10 lags):")
print(np.round(pacf_vals, 3))

# Fit ARMA(p,q) using best guess from PACF/ACF 
model = ARIMA(arma_data, order=(2, 0, 1))
result = model.fit()
print("\nFitted ARMA(2,1) model summary:")
print(result.summary())

# Residuals and diagnostics 
residuals = result.resid
fig, ax = plt.subplots(2, 1, figsize=(12, 5))
ax[0].plot(residuals)
ax[0].set_title("Residuals from ARMA(2,1) Model")
plot_acf(residuals, ax=ax[1], lags=30)
plt.tight_layout()
plt.show()