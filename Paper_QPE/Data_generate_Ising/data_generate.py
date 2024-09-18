""" Main routines for QCELS 

Quantum complex exponential least squares (QCELS) can be used to
estimate the ground-state energy with reduced circuit depth. 

Last revision: 11/22/2022
"""


import numpy as np
from matplotlib import pyplot as plt
import math
import source as srs

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['lines.markersize'] = 10

num_sites = 8
dim = 2**num_sites
J = 1.0
g = 3

ham = -J*srs.H_ZZ(num_sites) + g*srs.H_X(num_sites)
eig,vec = np.linalg.eig(ham)
spectrum = np.zeros(dim)
eigenstates = np.zeros((dim,dim),dtype = complex)
idx_sort = np.argsort(eig)
for i in range(dim):
    idx = idx_sort[i]
    spectrum[i] = eig[idx]
    eigenstates[:,i] = vec[:,idx]

spectrum = (math.pi/(4*srs.matrix_norm(ham,dim)))*spectrum


population = np.zeros(dim)

for i in range(10):
    population[i] = 1/4**i 
population[i] = population[i]/np.sum(population)

plt.plot(spectrum, population,'b-o');plt.show()

np.save('frequencies.npy',spectrum)
np.save('amplitudes.npy',population)