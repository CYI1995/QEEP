import cvxpy as cp
import numpy as np
import math
from matplotlib import pyplot as plt
import random


N = 1000
S = 50

# Noise = np.zeros(N,dtype = complex)

# # sigma = math.sqrt(2)/8
# sigma = 0.05
# X = np.random.normal(0,sigma,N)
# Y = np.random.normal(0,sigma,N)

# for i in range(N):
#     Noise[i] = X[i] + 1j*Y[i]

# np.save('noise_signal.npy',Noise)

Samples = np.zeros(S)

N1 = int(N/4)
N2 = int(3*N/4)

for i in range(S):
    # temp_t = np.random.randint(0,N-1)
    temp_t = np.random.randint(N1,N2)
    Samples[i] = temp_t

print(Samples)

np.save('sampled_times.npy',Samples)