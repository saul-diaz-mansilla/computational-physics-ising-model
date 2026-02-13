# computational-physics-ising-model
A brief work that covers the basics of the Ising model: Metropolis algorithm to obtain the magnetization and other macroscopic variables such as energy, specific heat and susceptibility. Finite scaling effects also studied. Part of a computational physics course in Spanish.

Read final report in "CA__Práctica_Tema_7.pdf".

Folders:
- "data_i": data stored from the simulations.
- "Figures": figures obtained from the data analysis.

Codes:
- "ising.py": Metropolis algorithm for fixed lattice size. Exports data to folder "data".
- "ising_L.py": Metropolis algorithm for variable lattice size. Exports data to folder "data_4". Main data for analysis.
- "ising_H.py": Metropolis algorithm with external magnetic field. Exports data to folder "data_3".
- "plots.py": Plots magnetization, energy, specific heat, susceptibility, specific heat for different lattice sizes, and entropy.
- "plots_2.py": Finite size scaling for specific heat and susceptibility.
- "magnetization.py": Legacy code testing 3d data storage.