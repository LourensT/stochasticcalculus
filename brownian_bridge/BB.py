# %%
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")

from sample_paths.BM import BM

def sample_mid_point(r, t, B_r, B_t):
    """
    r < t
    B_r, B_t \in \mathbb{R}, sample paths of Brownian motion at r, t

    return: tuple: (r+t)/2, sample path of Brownian motion at (r+t)/2
    """
    #midpoints
    assert r < t, "invalid times received"

    s = (r + t) / 2

    mean = ((t-s)*B_r + (s-r)*B_t)/(t-r)
    var= ((s-r)*(t-s))/(t-r)

    sample = np.random.normal(mean, np.sqrt(var))

    return s, sample

def BB_samplepath(T, delta_t, repeats):
    #sample paths of Brownian motion at 0, T
    B0 = BM(T, int(T/delta_t)-1, 1).transpose()[0]
    T0 = np.linspace(delta_t, T, int(T / delta_t), endpoint=True)

    # starting points
    paths = [B0]
    times = [T0]

    # refine grid
    for i in range(repeats):
        midpoints = np.ndarray(len(times[i]) - 1)
        midpoint_times = np.ndarray(len(times[i]) - 1)

        # sample midpoints
        for j in range(len(midpoints)):
            midpoint_times[j], midpoints[j] = (
                sample_mid_point(times[i][j], times[i][j+1], paths[i][j], paths[i][j+1])
            )

        paths.append(np.insert(paths[i],  list(range(1, len(midpoints)+1)), midpoints))
        times.append(np.insert(times[i],  list(range(1, len(midpoints)+1)), midpoint_times))

    return paths, times

def viz(paths, times):
    """draw BM sample paths"""
    for time, path in zip(times, paths):
        plt.plot(time, path, alpha=0.5)

    plt.title(f"{len(paths) -1} refinements of Brownian Bridge sample path")
    plt.show()

# %%
if __name__=="__main__":
    viz(*BB_samplepath(1, 1/10, 8))
