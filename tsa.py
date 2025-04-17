import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from statsmodels.tsa.arima.model import ARIMA

def timeSimulaton() -> pd.Series:
    
    """_summary_
    - This  method simulate a non-stationary time series with three
    important components (trend, seasons, and randomness (noise))

    Returns:
        pd.Series: _description_
    """
    t = np.arange(200) # time index
    trend = 0.05 * t # Slope
    seasonality = 2 * np.sin(2 * np.pi * t / 12) # Periods
    noise = np.random.normal(0, 1, 200) # random component

    # Simulated time series
    ts = trend + seasonality + noise # non-stationary ts
    ts_series = pd.Series(ts) 

    # Ploting time series
    plt.figure(figsize=(10, 4))
    plt.plot(ts_series)
    plt.title("Simulated Time Series: Trend + Seasonality + Noise")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

    # Ploting ACF and PACF
    _, ax = plt.subplots(2, 1, figsize=(10, 6))
    sm.graphics.tsa.plot_acf(ts_series, lags=30, ax=ax[0])
    sm.graphics.tsa.plot_pacf(ts_series, lags=30, ax=ax[1])
    plt.tight_layout()
    plt.show()
    
    return ts_series

# Fit AR(1)


if __name__ == "__main__":
    
    X_t = timeSimulaton() # Ploating and fetching the simulated series
    
    model_ar = ARIMA(X_t, order=(8, 0, 0)).fit() # AR(1)
    print(model_ar.summary())

    # Fit MA(1)
    model_ma = ARIMA(X_t, order=(0, 0, 5)).fit()
    print(model_ma.summary())

    # Fit ARMA(1,1)
    model_arma = ARIMA(X_t, order=(8, 0, 5)).fit()
    print(model_arma.summary())

    # Plot residuals
    model_arma.resid.plot(title='ARMA(1,1) Residuals', figsize=(10, 4))
    plt.grid()
    plt.show()
