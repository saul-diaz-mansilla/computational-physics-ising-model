import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

J = 1

def E(grid):
    energy = -J * (np.sum(grid * np.roll(grid, shift=1, axis=0)) +
                   np.sum(grid * np.roll(grid, shift=1, axis=1)))
    return energy

n_side = 20
n_steps = 250
n_T = 100
T_val = np.linspace(0.5, 3.5, n_T)

M_T = []
E_T = []
with pd.HDFStore(r"C:\Users\Usuario\Documents\Documentos\Universidad\Computacion_Avanzada\Tema_7\data_" + str(n_side) + "_" + str(n_steps) + "_3.h5", mode='r') as store:
    for i in range(n_T):
        loaded_df = store[f"spins_{i}"]
        spins = loaded_df.values
        M_T.append(np.abs(np.sum(spins) / n_side ** 2))
        E_T.append(E(spins) / n_side ** 2)

figure1 = plt.figure(figsize=(12, 5))
fig1 = figure1.add_subplot(121)
fig1.plot(T_val, M_T, '.')
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

plt.tight_layout()
plt.show()