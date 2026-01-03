import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.optimize import curve_fit
from scipy.integrate import quad
import pandas as pd

np.random.seed(42)  # Significado de la vida, el universo y todo lo demás

J = 1 # Interacción entre espines

def E(grid):
    energy = -J * (np.sum(grid * np.roll(grid, shift=1, axis=0)) +
                   np.sum(grid * np.roll(grid, shift=1, axis=1)))
    return energy

def MC_step_sistematic(grid, T):
    dE_n = 0
    side = len(grid)
    for i in range(side):
        for j in range(side):
            dE = 2 * J * grid[i,j] * (grid[(i+1) % side,j] + grid[(i-1) % side,j] + grid[i,(j+1) % side] + grid[i,(j-1) % side])
            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                dE_n += dE
                grid[i,j] *= -1
    return grid, dE_n

# Funciones de ajuste
def magnetization_crit(T, T_c, beta):
    return (T_c - T)**beta

def magnetization_2(T, beta):
    return (1 - np.sinh(2 / T)**(-4))**beta

def susceptibility(T, T_c, gamma, C1, C2):
    return (C1 * (T - T_c)**(-gamma)) * (T < T_c) + (C2 * (T - T_c)**(-gamma)) * (T > T_c)

def lorentzian(T, T_0, beta, A):
    T_reflected = 2 * np.sqrt(T_0**2 - 2*beta**2) - T
    return A / np.sqrt((T_0**2 - T_reflected**2)**2 + (2 * beta * T_reflected)**2)

def dSdT(T, T_0, beta, A):
    T_reflected = 2 * np.sqrt(T_0**2 - 2*beta**2) - T
    return A / np.sqrt((T_0**2 - T_reflected**2)**2 + (2 * beta * T_reflected)**2) / T

L_val = np.arange(5, 41, 1)
L_val = [int(L_val[i]) for i in range(len(L_val))]
C_max = []
chi_max = []
for n_side in L_val:
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
    chi = []
    S_T = []
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
        M_samples = np.array(M_samples)
        M_T.append(np.mean(M_samples))
        E_T.append(np.mean(E_samples) / n_side ** 2)
        C_N.append((np.mean(E_samples**2) - np.mean(E_samples)**2) / n_side**2 / T**2)
        chi.append((np.mean(M_samples**2) - np.mean(M_samples)**2) / n_side**2 / T)

    p_C1, c_C1 = curve_fit(lorentzian, T_val, C_N, p0=[T_center, 0.125, 1])
    T_c3, beta3, A = p_C1
    C_max.append(lorentzian(np.sqrt(T_c3**2 - 2*beta3**2), T_c3, beta3, A))
    chi_max.append(np.max(chi))

    for T in T_val:
        S, _ = quad(lambda T: dSdT(T, T_c3, beta3, A), T_0, T)
        S_T.append(S)

    filename = r"data_4\values_" + str(n_side) + ".csv"
    np.savetxt(filename, 
            np.column_stack((T_val, M_T, E_T, C_N, chi, S_T)),
            delimiter=',', 
            header="T,M,E,C,chi,S", 
            comments='',
            fmt='%.6f')

filename = r"data_4\L_var.csv"
np.savetxt(filename, 
           np.column_stack((L_val, C_max, chi_max)),
           delimiter=',', 
           header="L,C,chi", 
           comments='',
           fmt='%.6f')

figure3 = plt.figure()
fig3 = figure3.add_subplot(111)
fig3.plot(np.log(L_val), C_max, '.')
fig3.set_title(r"$C_{max}/N$ en función de $\log(L)$")
fig3.set_xlabel(r"$\log(L)")
fig3.set_ylabel(r"$C$")
fig3.grid()

plt.tight_layout()
plt.show()

