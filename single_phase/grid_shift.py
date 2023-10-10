import cvxpy as cp
import numpy as np
import math
from matplotlib import pyplot as plt
import random

# def Fourier_matrix(L):
#     F = np.zeros((L,L))
#     a = 2*math.pi/L 
#     for i in range(L):
#         for j in range(L):
#             aij = a*i*j 
#             # F[j][i] = (math.cos(aij) - 1j*math.sin(aij))/L
#             F[j][i] = math.cos(aij)/L

#     return F 

def matrix_dsum(A,B):

    dsum = np.zeros( np.add(A.shape,B.shape) )

    dsum[:A.shape[0],:A.shape[1]]=A
    dsum[A.shape[0]:,A.shape[1]:]=B 

    return dsum

def sample_single_value(t):

    p = (t+1)/2
    sample = random.uniform(0,1)
    if (sample < p):
        return 1
    else:
        return -1

# Generate a random non-trivial linear program.
Sample_number = 50
Total_length = 1000
T_samples = np.load('sampled_times.npy')
Hadamard = np.load('Hadamard_data.npy')
eps = 0.344*math.sqrt(Sample_number)

c1 = np.ones(Total_length)
zeros = np.zeros(Total_length)

k1 = 20.25
k2 = 100 
c1 = 0.9 
c2 = 0.1

F1 = np.zeros((Sample_number,Total_length))
F2 = np.zeros((Sample_number,Total_length))
F = np.zeros((Sample_number,Total_length),dtype = complex)

for i in range(Sample_number):
    temp_t = T_samples[i]
    for j in range(Total_length):
        F1[i][j] = math.cos(temp_t*2*math.pi*j/Total_length)
        F2[i][j] = (-1)*math.sin(temp_t*2*math.pi*j/Total_length)
        F[i][j] = F1[i][j] + 1j*F2[i][j]

M = 20
X_axis = np.zeros(M)
Y1_axis = np.zeros(M)
Y2_axis = np.zeros(M)
for m in range(M):

    shift = -0.5 + (m+1)/M


    y = np.zeros(Sample_number,dtype = complex)
    for i in range(Sample_number):
        temp_t = T_samples[i]
        p = -temp_t*2*math.pi*shift/Total_length
        y[i] = (Hadamard[i])*(math.cos(p) -1j*math.sin(p))

    x = cp.Variable(Total_length)
    constraint = [cp.norm(F @ x - y,2) <= eps]
    prob = cp.Problem(cp.Minimize(cp.norm(x,1)), constraint)
    prob.solve()

    temp_k_list = x.value
    max_idx = np.argsort(temp_k_list)[-1]
    # print(max_idx)
    k_solution = np.zeros(Total_length)
    k_solution[max_idx] = 1
    vec_diff = F.dot(k_solution) - y
    err = abs(np.vdot(vec_diff,vec_diff))


    X_axis[m] = shift 
    Y1_axis[m] = err
    Y2_axis[m] = (max_idx + shift)/Total_length

np.save('grid_shift.npy',X_axis)
np.save('error.npy',Y1_axis)
np.save('solution.npy',Y2_axis)
