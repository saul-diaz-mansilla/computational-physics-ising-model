import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.special import ellipk
import pandas as pd

np.random.seed(42)  # Significado de la vida, el universo y todo lo demás

J = 1 # Interacción entre espines

def E(grid):
    energy = -J * (np.sum(grid * np.roll(grid, shift=1, axis=0)) +
                   np.sum(grid * np.roll(grid, shift=1, axis=1)))
    return energy

def MC_step(grid, T):
    side = len(grid)
    for _ in range(side**2):
        i = np.random.randint(0, side - 1)
        j = np.random.randint(0, side - 1)
        dE_n = 0
        side = len(grid)
        dE = 2 * J * grid[i,j] * (grid[(i+1) % side,j] + grid[(i-1) % side,j] + grid[i,(j+1) % side] + grid[i,(j-1) % side])
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            dE_n += dE
            grid[i,j] *= -1
    return grid, dE_n

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

def energy(T):
    return -1/np.tanh(2/T) * (1 + 2/np.pi* (2*np.tanh(2/T)**2 - 1)*ellipk((2*np.sinh(2/T)/np.cosh(2/T)**2)**2))

def lorentzian(T, T_0, beta, A):
    T_reflected = 2 * np.sqrt(T_0**2 - 2*beta**2) - T
    return A / np.sqrt((T_0**2 - T_reflected**2)**2 + (2 * beta * T_reflected)**2)

def dSdT(T, T_0, beta, A):
    T_reflected = 2 * np.sqrt(T_0**2 - 2*beta**2) - T
    return A / np.sqrt((T_0**2 - T_reflected**2)**2 + (2 * beta * T_reflected)**2) / T

n_side = 20
n_steps = 250

# Define custom T values
T_0 = .5
T_n = 3.5
T_center = 2 / np.log(1 + np.sqrt(2))  # Critical temperature for the 2D Ising model
n_points = 100
T_dense = truncnorm.rvs((T_0 - T_center) / 0.3, (T_n - T_center) / 0.3, loc=T_center, scale=0.3, size=int(0.8 * n_points))
T_sparse = np.linspace(T_0, T_n, int(0.2 * n_points))
T_val = np.sort(np.concatenate((T_dense, T_sparse)))

# T_val = np.linspace(0.5, 3.5, 100)
# T_val = generate_truncated_normal_points(0.5, 3.5, 2.2, 0.5, 100)
M_T = []
E_T = []
C_N = []
chi = []
S_T = []
# with pd.HDFStore(r"C:\Users\Usuario\Documents\Documentos\Universidad\Computacion_Avanzada\Tema_7\data_" + str(n_side) + "_" + str(n_steps) + "_4.h5", mode='w') as store:
for n in range(len(T_val)):
    T = T_val[n]
    # More steps closer to T=2.2
    weight = np.exp(-((T - T_center) / 0.12)**2)  # Gaussian weighting
    n_steps = int(150 + 1000 * weight)

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

filename = r"C:\Users\Usuario\Documents\Documentos\Universidad\Computacion_Avanzada\Tema_7\data\values_" + str(n_side) + "_3.csv"
np.savetxt(filename, 
           np.column_stack((T_val, M_T, E_T, C_N, chi)),
           delimiter=',', 
           header="T,M,E,C,chi", 
           comments='',
           fmt='%.2f')

M_T = np.abs(M_T)

figure1 = plt.figure(figsize=(12, 5))

p_M1, c_M1 = curve_fit(magnetization_crit, T_val, M_T, p0=[T_center, 0.125])
T_c1, beta1 = p_M1
p_M2, c_M2 = curve_fit(magnetization_2, T_val, M_T, p0=[0.125])
beta2 = p_M2[0]

fig1 = figure1.add_subplot(121)
fig1.plot(T_val, M_T, '.', label="Datos numéricos")
fig1.plot(T_val, magnetization_crit(T_val, T_c1, beta1), label="Ajuste crítico")
fig1.plot(T_val, magnetization_2(T_val, beta2), label="Ajuste 2")
fig1.set_title("Magnetización en función de la temperatura")
fig1.set_xlabel(r"$T (J/k_B)$")
fig1.set_ylabel(r"$M (J)$")
fig1.grid()

fig1 = figure1.add_subplot(122)
fig1.plot(T_val, E_T, '.')
fig1.set_title("Energía en función de la temperatura")
fig1.set_xlabel(r"$T (J/k_B)$")
fig1.set_ylabel(r"$E (J)$")
fig1.grid()

p_C1, c_C1 = curve_fit(lorentzian, T_val, C_N, p0=[T_center, 0.125, 1])
T_c3, beta3, A = p_C1

figure2 = plt.figure(figsize=(12, 5))
fig3 = figure2.add_subplot(121)
fig3.plot(T_val, C_N, '.')
fig3.plot(T_val, lorentzian(T_val, T_c3, beta3, A), label="Ajuste lorentziano")
fig3.set_title("C en función de la temperatura")
fig3.set_xlabel(r"$T (J/k_B)$")
fig3.set_ylabel(r"$C$")
fig3.legend()
fig3.grid()

fig4 = figure2.add_subplot(122)
fig4.plot(T_val, chi, '.')
fig4.set_title(r"$\chi$ en función de la temperatura")
fig4.set_xlabel(r"$T (J/k_B)$")
fig4.set_ylabel(r"$\chi$")
fig4.grid()

for T in T_val:
        S, _ = quad(lambda T: dSdT(T, T_c3, beta3, A), T_0, T)
        S_T.append(S)

figure5 = plt.figure(figsize=(12, 5))
fig5 = figure5.add_subplot(111)
fig5.plot(T_val, S_T, '.')
fig5.set_title("Entropía en función de la temperatura")
fig5.set_xlabel(r"$T (J/k_B)$")
fig5.set_ylabel(r"$S$")
fig5.grid()

plt.tight_layout()
plt.show()
