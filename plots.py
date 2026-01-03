import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import ellipk
from scipy.integrate import quad

# Funciones de ajuste
def magnetization_crit(T, T_c, beta):
    return ((T_c - T)**beta) * (T < T_c)

def magnetization_2(T, T_c, beta):
    return ((1 - np.sinh(2 / T)**(-4))**beta) * (T < T_c)

# def susceptibility(T, T_c, gamma, C1, C2):
#     with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for invalid operations
#         below_Tc = (T < T_c) * (C1 * np.power(T_c - T, -gamma))
#         above_Tc = (T > T_c) * (C2 * np.power(T - T_c, -gamma))
#         return np.nan_to_num(below_Tc + above_Tc)  # Replace NaN or infinity with 0

def susceptibility(T, T_c, gamma, C):
    return C * (T - T_c)**(-gamma)

def lorentzian(T, T_0, beta, A):
    T_reflected = 2 * np.sqrt(T_0**2 - 2*beta**2) - T
    return A / np.sqrt((T_0**2 - T_reflected**2)**2 + (2 * beta * T_reflected)**2)

def gaussian(T, T_0, sigma, A):
    return A * np.exp(-0.5 * ((T - T_0) / sigma)**2)

def energy(T):
    return -1/np.tanh(2/T) * (1 + 2/np.pi* (2*np.tanh(2/T)**2 - 1)*ellipk((2*np.sinh(2/T)/np.cosh(2/T)**2)**2))

def integrand(T):
    dE_dT = np.gradient(energy(T_plot), T_plot)  # Compute the gradient of energy
    return np.interp(T, T_plot, dE_dT / T)  # Interpolate to get the value at T

n_side = 20
filename = r"data_4\values_" + str(n_side) + ".csv"
data = np.genfromtxt(filename, delimiter=',', skip_header=1)

T_center = 2 / np.log(1 + np.sqrt(2))  # Critical temperature for the 2D Ising model

# Split into separate arrays
T_val = data[:, 0]  # First column
M_T = data[:, 1]  # Second column
E_T = data[:, 2]  # Third column
C_N = data[:, 3]  # Fourth column
chi = data[:, 4]  # Fifth column
S = data[:, 5]  # Sixth column

M_T = np.abs(M_T)
T_plot = np.linspace(0.5, 3.5, 1000)

figure1 = plt.figure(figsize=(12, 5))

p_M1, c_M1 = curve_fit(magnetization_crit, T_val, M_T, p0=[T_center, 0.125])
T_c1, beta1 = p_M1
p_M2, c_M2 = curve_fit(magnetization_2, T_val, M_T, p0=[T_center, 0.125])
T_c2, beta2 = p_M2

fig1 = figure1.add_subplot(121)
fig1.plot(T_val, M_T, '.', label="Datos numéricos")
fig1.plot(T_plot, magnetization_crit(T_plot, T_c1, beta1), label="Campo medio")
fig1.plot(T_plot, magnetization_2(T_plot, T_c2, beta2), label="Onsager")
fig1.set_title("Magnetización en función de la temperatura")
fig1.set_xlabel(r"$T (J/k_B)$")
fig1.set_ylabel(r"$M$")
fig1.legend()
fig1.grid()

fig2 = figure1.add_subplot(122)
fig2.plot(T_val, E_T, '.', label="Datos numéricos")
fig2.plot(T_plot, energy(T_plot), label="Onsager")
fig2.set_title("Energía en función de la temperatura")
fig2.set_xlabel(r"$T (J/k_B)$")
fig2.set_ylabel(r"$E (J)$")
fig2.legend()
fig2.grid()

figure3 = plt.figure(figsize=(12, 5))
# figure3 = plt.figure()

p_C1, c_C1 = curve_fit(lorentzian, T_val, C_N, p0=[T_center, 0.125, 1])
T_c3, beta3, A = p_C1
p_C2, c_C2 = curve_fit(gaussian, T_val, C_N, p0=[T_center, 0.5, 1])
T_c4, sigma, A2 = p_C2

fig3 = figure3.add_subplot(121)
fig3.plot(T_val, C_N, '.', label="Datos numéricos")
fig3.plot(T_plot, lorentzian(T_plot, T_c3, beta3, A), label="Ajuste lorentziano")
fig3.plot(T_plot, np.gradient(energy(T_plot), T_plot), label="Onsager")
fig3.set_title("Calor específico en función de la temperatura")
fig3.set_xlabel(r"$T (J/k_B)$")
fig3.set_ylabel(r"$C$")
fig3.legend()
fig3.grid()

# Print adjusted parameters
print("Adjusted Parameters:")
print(f"Magnetization (Campo medio): T_c1 = {T_c1:.3f}, beta1 = {beta1:.3f}")
print(f"Magnetization (Onsager): T_c2 = {T_c2:.3f}, beta2 = {beta2:.3f}")
print(f"Specific Heat (Lorentzian): T_c3 = {T_c3:.3f}, beta3 = {beta3:.3f}, A = {A:.3f}")
print(f"Maximum Temperature: {np.sqrt(T_c3**2 - 2*beta3**2):.3f}")

# p_s, c_s = curve_fit(susceptibility, T_val, chi, p0=[T_center, 7/4, 1e-6])
# T_c5, gamma, C = p_s

fig4 = figure3.add_subplot(122)
fig4.plot(T_val, chi, '.', label="Datos numéricos")
# fig4.plot(T_plot, susceptibility(T_plot, T_c5, gamma, C), label="Ajuste")
fig4.set_title(r"$\chi$ en función de la temperatura")
fig4.set_xlabel(r"$T (J/k_B)$")
fig4.set_ylabel(r"$\chi$")
fig4.grid()

figure5 = plt.figure()
fig5= figure5.add_subplot(111)

for n_side in [10, 20, 30, 40]:
    filename = r"data_2\values_" + str(n_side) + ".csv"
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    T_val = data[:, 0]  # First column
    C_max = data[:, 3]  # Fourth column
    
    p_C1, c_C1 = curve_fit(lorentzian, T_val, C_N, p0=[T_center, 0.125, 1])
    T_c3, beta3, A = p_C1

    # fig5.plot(T_val, C_max, '.', label=f"L={n_side}")
    fig5.plot(T_plot, lorentzian(T_plot, T_c3, beta3, A), label=f"L={n_side}")

# fig5.plot(T_plot, np.gradient(energy(T_plot), T_plot), label="Onsager")
fig5.set_title("Calor específico en función de la temperatura")
fig5.set_xlabel(r"$T (J/k_B)$")
fig5.set_ylabel(r"$C$")
fig5.legend()
fig5.grid()

# S_real = [quad(integrand, T_plot[0], T)[0] for T in T_plot]

figure6 = plt.figure()
fig6 = figure6.add_subplot(111)
fig6.plot(T_val, S, '.', label="Datos numéricos")
# fig6.plot(T_plot, S_real, label="Integral numérica")
fig6.set_title("Entropía en función de la temperatura")
fig6.set_xlabel(r"$T (J/k_B)$")
fig6.set_ylabel(r"$S$")
fig6.legend()
fig6.grid()

plt.tight_layout()

plt.show()
