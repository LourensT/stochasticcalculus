# %%
from scipy import stats
import numpy as np

def BM(T, K, M):
    """
    T > 0
    K,M \in \mathbb{N}

    return: (K+1)xM matrix containing M realizations of Brownian motion sample paths
    """

    t = np.linspace(0, T, K+1, endpoint=True)

    A = np.ndarray((K+1, M))
    A[0,:] = 0
    for i in range(1, len(t)):
        A[i,:] = A[i-1,:] + stats.norm.rvs(scale=np.sqrt(t[i] - t[i-1]), size=M)

    return A 

import matplotlib.pyplot as plt
def viz(A, title=None):
    """draw BM sample paths"""
    t = np.linspace(0, 1, A.shape[0], endpoint=True)
    for path in A.T:
        plt.plot(t, path)
    
    if title is not None:
        plt.title(title)

if __name__=="__main__":
    A= BM(1, 10**5, 5)
    viz(A)
