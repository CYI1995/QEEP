""" Main routines for QCELS 

Quantum complex exponential least squares (QCELS) can be used to
estimate the ground-state energy with reduced circuit depth. 

Last revision: 11/22/2022
"""

import scipy.io as sio
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from scipy.special import erf
from qcels import *

if __name__ == "__main__":
    import scipy.io as sio
    import numpy as np
    from copy import deepcopy
    from scipy.optimize import minimize
    from matplotlib import pyplot as plt
    from scipy.special import erf
    from mpl_toolkits.mplot3d import Axes3D
    import math
    import matplotlib
    import source as srs
    matplotlib.rcParams['font.size'] = 15
    matplotlib.rcParams['lines.markersize'] = 10

    num_sites = 8
    dim = 2**num_sites
    J = 1.0
    g = 4
    
    num_eigenstates_max = 100
    
    ham0 = -J*srs.H_ZZ(num_sites) + 1*srs.H_X(num_sites)
    eig,vec = np.linalg.eig(ham0)
    gs_idx = np.argsort(eig)[0]
    init_state = np.zeros(dim)
    for i in range(dim):
        init_state[i] = vec[i][gs_idx]

    ham = -J*srs.H_ZZ(num_sites) + g*srs.H_X(num_sites)
    eig,vec = np.linalg.eig(ham)
    eigenenergies = np.zeros(dim)
    eigenstates = np.zeros((dim,dim),dtype = complex)
    idx_sort = np.argsort(eig)
    for i in range(dim):
        idx = idx_sort[i]
        eigenenergies[i] = eig[idx]
        eigenstates[:,i] = vec[:,idx]

    ground_state = eigenstates[:,0]

    population_raw = np.abs(np.dot(eigenstates.conj().T, init_state))**2

    spectrum, population = generate_spectrum_population(eigenenergies, population_raw, 
                                                    [population_raw[0]])

    population[0] = 0.7 
    population[1] = 0.2 
    for i in range(2,dim):
        population[i] = 1/2540
    
    plt.plot(spectrum, population,'b-o');plt.show()

    np.save('frequencies.npy',spectrum)
    np.save('amplitudes.npy',population)