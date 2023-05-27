# %%
# SDE in question: 
# $dXt = − sin(X_t)\cos^{3}(X_t)dt + \cos^{2}(X_t)dB_t$
# has strong solution $X_t = arctan(B_t + tan(x_0))$

import sys
sys.path.append("../")
from sample_paths.BM import BM, viz

import numpy as np
import matplotlib.pyplot as plt

def EM(X_0, B, h):
    """
    Euler-Maruyama scheme for  SDE
    """
    X = np.ndarray(B.shape)
    X[0,:] = X_0
    for i in range(1, B.shape[0]):
        X[i,:] = (
            X[i-1,:] 
            - np.sin(X[i-1,:])*(np.cos(X[i-1,:])**3)*h 
            + np.cos(X[i-1,:])**2*(B[i,:] - B[i-1,:])
        )

    return X

def Milstein(X_0, B, h):
    """
    Milstein scheme for SDE
    """
    X = np.ndarray(B.shape)
    X[0,:] = X_0
    for i in range(1, B.shape[0]):
        X[i,:] = (
            X[i-1,:] 
            - np.sin(X[i-1,:])*(np.cos(X[i-1,:])**3)*h 
            + np.cos(X[i-1,:])**2*(B[i,:] - B[i-1,:]) 
            - 0.5*np.sin(X[i-1,:]) * np.cos(X[i-1,:])**2*(B[i,:] - B[i-1,:])**2
        )
    
    return X

# parameters
T = x_0 = 1
l = np.array([0, 1, 2, 3, 4, 5])
N_l = 10*(2**l)
M = 10**5

# get M instances of the strong solution

# for each l, get the error of both EM and Milstein schemes
EM_SE = []
EM_RMSE = []
Milstein_SE = []
Milstein_RMSE = []

for N in N_l:
    h = T/N
    B = BM(T, N, M)

    X_T = np.arctan(B[-1,:] + np.tan(x_0))

    # Euler-Maruyama approximation
    X_em = EM(x_0, B, h)
    EM_SE.append(np.mean(np.abs(X_T - X_em[-1,:])))
    EM_RMSE.append(np.sqrt(np.mean((X_T - X_em[-1,:])**2)))

    # Millstein approximation
    X_milstein = Milstein(x_0, B, h)
    Milstein_SE.append(np.mean(np.abs(X_T - X_milstein[-1,:])))
    Milstein_RMSE.append(np.sqrt(np.mean((X_T - X_milstein[-1,:])**2)))

# %%
h_l = T/N_l
# plot EM_SE against h_l in loglog
plt.loglog(h_l, EM_SE, label="Euler-Maruyama, standard error", marker="o")
plt.gca().invert_xaxis()

EM_SE_fit = np.polyfit(np.log(h_l), np.log(EM_SE), 1)
x = np.linspace(min(h_l), max(h_l), 100)
plt.plot(x, np.exp(EM_SE_fit[1])*x**EM_SE_fit[0], label="best fit, slope={:.2f}".format(EM_SE_fit[0]), alpha=0.5)

plt.xlabel("$h_\mathcal{l}$")
plt.ylabel("$E_{M}(\mathcal{E}_s^{h_\mathcal{l}})$", size=14)
plt.legend()
plt.title("Euler-Maruyama, standard error")
plt.savefig("EM_SE.png", dpi=300)

# %%
# plot EM_RMSE against h_l in loglog
plt.figure()
plt.loglog(h_l, EM_RMSE, label="Euler-Maruyama, RMSE", marker="o")
plt.gca().invert_xaxis()

EM_RMSE_fit = np.polyfit(np.log(h_l), np.log(EM_RMSE), 1)
x = np.linspace(min(h_l), max(h_l), 100)
plt.plot(x, np.exp(EM_RMSE_fit[1])*x**EM_RMSE_fit[0], label="best fit, slope={:.2f}".format(EM_RMSE_fit[0]), alpha=0.5)

plt.xlabel("$h_\mathcal{l}$")
plt.ylabel("$E_{M}((\mathcal{E}_s^{h_\mathcal{l}})^2)^{1/2}$", size=14)
plt.legend()
plt.title("Euler-Maruyama, RMSE")
plt.savefig("EM_RMSE.png", dpi=300)

# %%
# plot Milstein_SE against h_l in loglog
plt.loglog(h_l, Milstein_SE, label="Milstein, standard error", marker="o")
plt.gca().invert_xaxis()

Milstein_SE_fit = np.polyfit(np.log(h_l), np.log(Milstein_SE), 1)
x = np.linspace(min(h_l), max(h_l), 100)
plt.plot(x, np.exp(Milstein_SE_fit[1])*x**Milstein_SE_fit[0], label="best fit, slope={:.2f}".format(Milstein_SE_fit[0]), alpha=0.5)

plt.xlabel("$h_\mathcal{l}$")
plt.ylabel("$E_{M}(\hat{\mathcal{E}}_s^{h_\mathcal{l}})$", size=14)
plt.legend()
plt.title("Milstein, standard error")
plt.savefig("Milstein_SE.png", dpi=300)

# %%
# plot Milstein_RMSE against h_l in loglog
plt.figure()
plt.loglog(h_l, Milstein_RMSE, label="Milstein, RMSE", marker="o")
plt.gca().invert_xaxis()

Milstein_RMSE_fit = np.polyfit(np.log(h_l), np.log(Milstein_RMSE), 1)
x = np.linspace(min(h_l), max(h_l), 100)
plt.plot(x, np.exp(Milstein_RMSE_fit[1])*x**Milstein_RMSE_fit[0], label="best fit, slope={:.2f}".format(Milstein_RMSE_fit[0]), alpha=0.5)

plt.xlabel("$h_\mathcal{l}$")
plt.ylabel("$E_{M}((\hat{\mathcal{E}}_s^{h_\mathcal{l}})^2)^{1/2}$", size=14)
plt.legend()
plt.title("Milstein, RMSE")
plt.savefig("Milstein_RMSE.png", dpi=300)
# %%

# weak error

T = x_0 = 1
l = np.array([0,1,2,3])
N_l = 4*(2**l)
h_l = T/N_l
M = 10**4
K = 10

def f(x):
    return max([x-1.1, 0])

expected_f_X = np.mean([f(i) for i in BM(T, 1, 10**7)[-1,:]])

EM_meanweakerror = []
Milstein_meanweakerror = []

for N in N_l:
    print(N)
    EM_weakerror = []
    Milstein_weakerror = []
    for _ in range(K):
        h = T/N
        B = BM(T, N, M)

        # Euler-Maruyama approximation
        X_em = EM(x_0, B, h)
        EM_weakerror.append(np.mean(np.abs(expected_f_X - np.mean([f(i) for i in X_em[-1,:]]))))

        # Millstein approximation
        X_milstein = Milstein(x_0, B, h)
        Milstein_weakerror.append(np.mean(np.abs(expected_f_X - np.mean([f(i) for i in Milstein(x_0, B, h)[-1,:]]))))

    EM_meanweakerror.append(np.mean(EM_weakerror))
    Milstein_meanweakerror.append(np.mean(Milstein_weakerror))

# %%

# %%
# plot Milstein_WSE against h_l in loglog
plt.loglog(h_l, Milstein_meanweakerror, label="Milstein, standard error", marker="o")
plt.gca().invert_xaxis()

Milstein_WSE_fit = np.polyfit(np.log(h_l), np.log(Milstein_meanweakerror), 1)
x = np.linspace(min(h_l), max(h_l), 100)
plt.plot(x, np.exp(Milstein_WSE_fit[1])*x**Milstein_WSE_fit[0], label="best fit, slope={:.2f}".format(Milstein_WSE_fit[0]), alpha=0.5)

plt.xlabel("$h_\mathcal{l}$")
plt.ylabel("$E_{K}(\hat{\mathcal{E}}_w^{h_\mathcal{l}})$", size=14)
plt.legend()
plt.title("Milstein, weak error")
plt.savefig("Milstein_WSE.png", dpi=300)

# %%
#  plot EM_WSE against h_l in loglog
plt.loglog(h_l,EM_meanweakerror, label="Euler-Maruyama, weak error", marker="o")
plt.gca().invert_xaxis()

EM_WSE_fit = np.polyfit(np.log(h_l), np.log(EM_meanweakerror), 1)
x = np.linspace(min(h_l), max(h_l), 100)
plt.plot(x, np.exp(EM_WSE_fit[1])*x**EM_WSE_fit[0], label="best fit, slope={:.2f}".format(EM_WSE_fit[0]), alpha=0.5)

plt.xlabel("$h_\mathcal{l}$")
plt.ylabel("$E_{K}(\mathcal{E}_w^{h_\mathcal{l}})$", size=14)
plt.legend()
plt.title("Euler-Maruyama, weak error")
plt.savefig("EM_WSE.png", dpi=300)
# %%
