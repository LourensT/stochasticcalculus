# %%
import sys
sys.path.append("../")
from sample_paths.BM import BM, viz

import numpy as np

def stock_sample_paths(T, c, sigma, S_0, M, delta_t):
    """
    T > 0
    c, sigma, S_0 \in \mathbb{R}
    M \in \mathbb{N}
    delta_t \in \mathbb{R}

    We model a stock price S_t = S_0 exp(ct + sigma B_t) where B_t is a Brownian motion

    return: (T/delta_t)xM matrix containing M realizations of stock sample paths
    """

    # sample paths of Brownian motion at 0, T
    B = BM(T, int(T/delta_t)-1, M)
    times = np.linspace(delta_t, T, int(T / delta_t), endpoint=True)

    print("paths are sampled")

    # calculate drifted brownian motion X_t = c t + sigma B_t
    X = np.ndarray(B.shape)
    X[0,:] = 0
    for i in range(1, len(times)):
        X[i,:] = X[i-1,:] + c*(times[i] - times[i-1]) + sigma*(B[i,:] - B[i-1,:])

    # calculate stock price S_t = S_0 exp(X_t)
    S = S_0 * np.exp(X)

    return S, times 

def expected_profit(paths, K_strike):
    """
    paths: (T/delta_t)xM matrix containing M realizations of stock sample paths
    K_strike: strike price

    return: expected profit
    """
    return np.mean(np.maximum(paths[-1,:] - K_strike, 0))

# %%
if __name__ == "__main__":
    S, times = stock_sample_paths(1, -0.125, 0.5, 10, 11, (1/10)*(2)**(-7))
    viz(S[:, :10], title="10 stock sample paths")
    print(f"montecarlo approximation of expected return on option: {expected_profit(S, 11)}")
# %%
