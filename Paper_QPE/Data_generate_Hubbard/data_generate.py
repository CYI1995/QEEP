

import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import scipy
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib
import source as mycode

matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['lines.markersize'] = 10

N = 4
dim = 2**(2*N)

t = 1
U = 10

H = U*mycode.hubbard_energy(N)
for i in range(N-1):
    H = H -t* mycode.hubbard_hopping(i,i+1,0,N) - mycode.hubbard_hopping(i,i+1,1,N)

H = H - t*mycode.hubbard_hopping_boundary(0,N) - t*mycode.hubbard_hopping_boundary(1,N)

eig,vec = scipy.linalg.eig(H)

spectrum = np.zeros(dim)
population = np.zeros(dim)
eig_sorted = np.argsort(eig)

norm = mycode.matrix_norm(H,dim)
scale_para = math.pi/(4*norm)

for l in range(dim):
    spectrum[l] = scale_para*eig[eig_sorted[l]].real


for i in range(10):
    population[i] = 1/4**i 
population[i] = population[i]/np.sum(population)

plt.plot(spectrum, population,'b-o');plt.show()

np.save('frequencies.npy',spectrum)
np.save('amplitudes.npy',population)