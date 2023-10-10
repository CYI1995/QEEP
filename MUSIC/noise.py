import cvxpy as cp
import numpy as np
import math
import source as srs

N = 201
M = 50
T = np.zeros(M)
for m in range(M):
    T[m] = np.random.randint(0,N-1)

print(T)

np.save('sampled_times.npy',T)