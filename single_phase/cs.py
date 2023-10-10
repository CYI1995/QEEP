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
Noise = np.load('noise_signal.npy')
Samples = np.load('sampled_times.npy')
eps = math.sqrt(Sample_number)*0.1

c1 = np.ones(Total_length)
zeros = np.zeros(Total_length)

k_star = 20.25
grid_shift = 0.25

F1 = np.zeros((Sample_number,Total_length))
F2 = np.zeros((Sample_number,Total_length))
F = np.zeros((Sample_number,Total_length),dtype = complex)

for i in range(Sample_number):
    # temp_t = np.random.randint(0,Total_length-1)
    temp_t = Samples[i]
    for j in range(Total_length):
        F1[i][j] = math.cos(temp_t*2*math.pi*j/Total_length)
        F2[i][j] = (-1)*math.sin(temp_t*2*math.pi*j/Total_length)
        F[i][j] = F1[i][j] + 1j*F2[i][j]
    
y = np.zeros(Sample_number,dtype = complex)
for i in range(Sample_number):
    temp_t = Samples[i]
    a = temp_t*2*math.pi*k_star/Total_length
    p = -temp_t*2*math.pi*grid_shift/Total_length
    y[i] = (math.cos(a)  -1j*math.sin(a) + Noise[i])*(math.cos(p) -1j*math.sin(p))

x = cp.Variable(Total_length)
constraint = [cp.norm(F @ x - y,2) <= eps]
# constraint = [F @ x == y]
prob = cp.Problem(cp.Minimize(cp.norm(x,1)), constraint)
prob.solve()

temp_k_list = x.value
max_idx = np.argsort(temp_k_list)[-1]
k_sol = max_idx + grid_shift

X1_axis = np.arange(Total_length)
Y1_axis = np.zeros(Total_length)
X2_axis = np.zeros(Sample_number)
Y2_axis = np.zeros(Sample_number)
for i in range(Total_length):
    Y1_axis[i] = math.cos(2*math.pi*k_sol*i/Total_length)

for j in range(Sample_number):
    X2_axis[j] = Samples[j]
    Y2_axis[j] = math.cos(2*math.pi*k_star*Samples[j]/Total_length)

plt.plot(X1_axis,Y1_axis, color = 'k')
plt.scatter(X2_axis,Y2_axis, color = 'r')
plt.show()

