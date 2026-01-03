import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.optimize import curve_fit
import pandas as pd

np.random.seed(42)  # Significado de la vida, el universo y todo lo demás

J = 1 # Interacción entre espines
mu = 1 # Momento magnético

def MC_step_sistematic(grid, T):
    dE_n = 0
    side = len(grid)
    for i in range(side):
        for j in range(side):
            dE = 2 * J * grid[i,j] * (grid[(i+1) % side,j] + grid[(i-1) % side,j] + grid[i,(j+1) % side] + grid[i,(j-1) % side]) + 2 * mu * H * grid[i,j]
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                dE_n += dE
                grid[i,j] *= -1
    return grid, dE_n

# Funciones de ajuste
def magnetization_crit(T, T_c, beta):
    return (T_c - T)**beta

def magnetization_2(T, beta):
    return (1 - np.sinh(2 / T)**(-4))**beta

def lorentzian(T, T_0, beta, A):
    T_reflected = 2 * np.sqrt(T_0**2 - 2*beta**2) - T
    return A / np.sqrt((T_0**2 - T_reflected**2)**2 + (2 * beta * T_reflected)**2)

n_side = 20
H_val = np.linspace(0, 1, 25)
Tcs = []

for H in H_val:
    # Define custom T values
    T_0 = .5
    T_n = 3.5
    T_center = 2 / np.log(1 + np.sqrt(2))  # Critical temperature for the 2D Ising model
    n_points = 100
    T_dense = truncnorm.rvs((T_0 - T_center) / 0.1, (T_n - T_center) / 0.1, loc=T_center, scale=0.1, size=int(0.8 * n_points))
    T_sparse = np.linspace(T_0, T_n, int(0.2 * n_points))
    T_val = np.sort(np.concatenate((T_dense, T_sparse)))

    # T_val = np.linspace(0.5, 3.5, 100)
    # T_val = generate_truncated_normal_points(0.5, 3.5, 2.2, 0.5, 100)
    M_T = []
    E_T = []
    C_N = []
    for n in range(len(T_val)):
        T = T_val[n]
        # More steps closer to T=2.2
        weight = np.exp(-((T - T_center) / 0.4)**2)  # Gaussian weighting
        n_steps = int(250 + 2000 * weight)      # base 150, up to ~2000 near T=2.2

        E_n = -2 * n_side**2 * J
        spins = np.ones((n_side,n_side))
        M_samples = []
        E_samples = []
        for _ in range(n_steps):
            spins, dE_n = MC_step_sistematic(spins, T)
            E_n += dE_n
            E_samples.append(E_n)
            M_samples.append(np.sum(spins) / n_side ** 2)
        E_samples = np.array(E_samples)
        M_T.append(np.mean(M_samples))
        E_T.append(np.mean(E_samples) / n_side ** 2)
        C_N.append((np.mean(E_samples**2) - np.mean(E_samples)**2) / n_side**2 / T**2)
    
    filename = r"data_3\values_" + str(H) + ".csv"
    np.savetxt(filename, 
            np.column_stack((T_val, M_T, E_T, C_N)),
            delimiter=',', 
            header="T,M,E,C", 
            comments='',
            fmt='%.2f')

    p_M1, c_M1 = curve_fit(magnetization_2, T_val, M_T, p0=[T_center, 0.125])
    T_c1, beta1 = p_M1
    Tcs.append(T_c1)

filename = r"data_3\H_var.csv"
np.savetxt(filename, 
           np.column_stack((H_val, Tcs)),
           delimiter=',', 
           header="L,C", 
           comments='',
           fmt='%.2f')

figure3 = plt.figure()
fig3 = figure3.add_subplot(111)
fig3.plot(H_val, Tcs, '.')
fig3.set_title(r"$T_c$ en función de $H$")
fig3.set_xlabel(r"$H (J/\mu)$")
fig3.set_ylabel(r"$T_c\ (J/k_B)$")
fig3.grid()

plt.tight_layout()
plt.show()

