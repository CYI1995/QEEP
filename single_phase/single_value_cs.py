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
    
def Hamadard_test(p,M):

    val = 0
    for m in range(M):
        r = random.uniform(0,1)
        if(r < p):
            val = val + 1
        else:
            val = val - 1

    return val/M

# Generate a random non-trivial linear program.
Sample_number = 50
Total_length = 1000
T_samples = np.load('sampled_times.npy')
Hadamard = np.load('Hadamard_data.npy')

eps = math.sqrt(Sample_number)*0.05
# eps = 0

c1 = np.ones(Total_length)
zeros = np.zeros(Total_length)

k_star = 20

F1 = np.zeros((Sample_number,Total_length))
F2 = np.zeros((Sample_number,Total_length))
F = np.zeros((Sample_number,Total_length),dtype = complex)

for i in range(Sample_number):
    temp_t = T_samples[i] 
    for j in range(Total_length):
        F1[i][j] = math.cos(temp_t*2*math.pi*j/Total_length)
        F2[i][j] = (-1)*math.sin(temp_t*2*math.pi*j/Total_length)
        F[i][j] = F1[i][j] + 1j*F2[i][j]

x = cp.Variable(Total_length)
constraint = [cp.norm(F @ x - Hadamard,2) <= eps]
prob = cp.Problem(cp.Minimize(cp.norm(x,1)), constraint)
prob.solve()
temp_k_list = x.value
max_idx = np.argsort(temp_k_list)[-1]
print(max_idx)

X1 = np.zeros(Sample_number)
Y1 = np.zeros(Sample_number)
X2 = np.zeros(Total_length)
Y2 = np.zeros(Total_length)

for i in range(Sample_number):
    X1[i] = T_samples[i]
    Y1[i] = Hadamard[i].real 

for j in range(Total_length):
    X2[j] = j 
    Y2[j] = math.cos(2*math.pi*j*k_star/Total_length)

plt.scatter(X1,Y1,c = "r")
plt.plot(X2,Y2,c="k")
plt.show()

