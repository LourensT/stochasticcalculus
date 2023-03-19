# %%
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from sample_paths.BM import BM

def DiscreteGBM(T, mu, sigma, x0, B):
    """
    Approximate Geometric Brownian Motion with discrete scheme
    """
    Z = np.ndarray(B.shape)
    n = B.shape[0]
    Z[0,:] = x0
    for i in range(1, B.shape[0]):
        Z[i,:] = Z[i-1,:] + mu*Z[i-1,:]*((T/n)) + sigma*Z[i-1, :]*(B[i,:] - B[i-1,:])

    return Z

if __name__=="__main__":
    # Test performance of DiscreteGBM compared to exact solution

    # GBM parameters
    mu = 0.5
    sigma = 1
    x0 = 2
    T = 1

    # experiment parameters
    l = np.array([0, 1, 2, 3, 4]) 
    N = 10*(2**l) # number of time steps
    M = 10**5
    h_l = T/N

    error = []
    for i in range(len(l)):
        B = BM(T, N[i], M)
        # approximate solution
        Z = DiscreteGBM(T, mu, sigma, x0, B)
        # exact solution
        X = x0 * np.exp((mu - 1/2*(sigma**2))*T + sigma*B[-1,:])

        # error
        error.append(np.mean(np.abs(X - Z[-1,:])))

    # %%
    fit = np.polyfit(np.log(h_l), np.log(error), 1)
    x = np.linspace(h_l[0], h_l[-1], 100)
    y = np.exp(fit[1])*x**fit[0]
    plt.scatter(h_l, error, label="Experimental Error")
    plt.plot(x, y, label=f"Fit, $r = {round(fit[0], 3)}$", color="orange")
    plt.ylabel("Error")
    plt.xlabel("h")
    plt.title("Experimental strong convergence rate")
    plt.legend()
