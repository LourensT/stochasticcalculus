# %%
from scipy import stats
import numpy as np

def PP(lam, T, K, M):
    """ 
    lam, T > 0
    K, M \in \mathbb{N}

    return A: (K+1)xM matrix
    """

    t = np.linspace(0, T, K+1, endpoint=True)

    A = np.ndarray((K+1, M))
    A[0,:] = 0
    for i in range(1, len(t)):
        A[i,:] = A[i-1,:] + stats.poisson.rvs(mu=lam*(t[i] - t[i-1]), size=M)

    return A    

if __name__== "__main__":
    A = PP(2.5, 1, 10**5, 5)

    from BM import viz
    viz(A)
