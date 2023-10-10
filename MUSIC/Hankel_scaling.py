import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import qeep_prony as prony
import source as srs


M = 100
X_axis = np.zeros(M)
Y_axis = np.zeros(M)


for m in range(M):
    N = 10 + 2*m
    k0 = 0.128
    k1 = 0.129
    v0 = np.zeros(N,dtype = complex)
    v1 = np.zeros(N,dtype = complex)
    for j in range(N):
        v0[j] = math.cos(2*math.pi*k0*j) + math.sin(2*math.pi*k0*j)
        v1[j] = math.cos(2*math.pi*k1*j) + math.sin(2*math.pi*k1*j)

    L = int(N/2)
    H0 = prony.Hankel(v0,0,L)
    H1 = prony.Hankel(v1,0,L)

    X_axis[m] = N 
    Y_axis[m] = srs.matrix_norm(H0 - H1, L)

plt.scatter(X_axis,Y_axis)
# plt.loglog()
plt.show()
