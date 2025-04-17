import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Simulate white noise
def wn_Gaussian(seed: np.ndarray = np.random.seed(29201))->None:
    """_summary_
    This function simulate a white noise time series problem from the 
    Gaussian distribution. For simplicity we assume the standard Gaussian,
    but any variance produces the same results.
    - This sequence is a completly random process. There is no any signficance
    ACF or PACF at any lag.

    Args:
        seed (np.ndarray, optional): _description_. Defaults to np.random.seed(29201).
    """
    # Simulate the white noise sequence (time-series) with length 200.
    seed = seed
    X_t: np.ndarray = np.random.normal(loc=0, scale=1, size=200)

    # Plot white noise
    plt.figure(figsize=(12, 6))
    plt.plot(X_t, label='White Noise')
    plt.title('Simulated White Noise Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # Computing and plotting the ACF and PACF for the white noise
    _, ax = plt.subplots(2, 1, figsize=(12, 6))
    sm.graphics.tsa.plot_acf(X_t, lags=30, ax=ax[0])
    sm.graphics.tsa.plot_pacf(X_t, lags=30, ax=ax[1])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    wn_Gaussian()