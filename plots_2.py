import numpy as np
import matplotlib.pyplot as plt


filename = r"C:\Users\Usuario\Documents\Documentos\Universidad\Computacion_Avanzada\Tema_7\data_4\L_var.csv"
data = np.genfromtxt(filename, delimiter=',', skip_header=1)
L_val = data[:, 0]
C_max = data[:, 1]
chi_max = data[:, 2]

# L_val = np.arange(5, 41, 1)
# chi_max = chi_max * L_val**2

m,n = np.polyfit(np.log(L_val), C_max, 1)
print(m,n)

figure3 = plt.figure()
fig3 = figure3.add_subplot(111)
fig3.plot(np.log(L_val), C_max, '.', label="Datos numéricos")
fig3.plot(np.log(L_val), m * np.log(L_val) + n, label="Ajuste lineal")
fig3.set_title(r"$C_{max}/N$ en función de $\log(L)$")
fig3.set_xlabel(r"$\log(L)$")
fig3.set_ylabel(r"$C$")
fig3.legend()
fig3.grid()

m2, n2 = np.polyfit(np.log(L_val), np.log(chi_max), 1)
print(m2,n2)

figure4 = plt.figure()
fig4 = figure4.add_subplot(111)
fig4.plot(np.log(L_val), np.log(chi_max), '.', label="Datos numéricos")
fig4.plot(np.log(L_val), m2 * np.log(L_val) + n2, label="Ajuste lineal")
fig4.set_title(r"$\chi_{max}/N$ en función de $\log(L)$")
fig4.set_xlabel(r"$\log(L)$")
fig4.set_ylabel(r"$\log(\chi_{max}/N)$")
fig4.legend()
fig4.grid()

plt.show()