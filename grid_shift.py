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

def Hamadard_test(p,M):

    val = 0
    for m in range(M):
        r = random.uniform(0,1)
        if(r < p):
            val = val + 1
        else:
            val = val - 1

    return val/M

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
Sample_number = 20
Total_length = 200

Batch = 100
Output_Data = np.zeros(Batch)
Total_Time = np.zeros(Batch)

theta = math.sqrt(math.pi)/(2*math.pi)
k1 = Total_length*theta
k2 = Total_length*0.5
c1 = 0.9 
c2 = 0.1
Ns = 100

for count in range(Batch):

    Samples = np.zeros(Sample_number)
    for i in range(Sample_number):
        temp_t = np.random.randint(0,Total_length-1)
        Samples[i] = temp_t

    Hadamard = np.zeros(Sample_number,dtype = complex)
    for i in range(Sample_number):

        t_temp = Samples[i]
        a1 = 2*math.pi*t_temp*k1/Total_length
        a2 = 2*math.pi*t_temp*k2/Total_length
        real = c1*math.cos(a1) + c2*math.cos(a2)
        imag = c1*math.sin(a1) + c2*math.sin(a2)
        real2 = Hamadard_test(0.5*(1+real),Ns)
        imag2 = Hamadard_test(0.5*(1+imag),Ns)
        Hadamard[i] = real2 - 1j*imag2

    T_tol = Ns*np.sum(Samples)
    Total_Time[count] = T_tol

    eps = 3*math.sqrt(Sample_number/50)

    # c1 = np.ones(Total_length)
    # zeros = np.zeros(Total_length)

    F1 = np.zeros((Sample_number,Total_length))
    F2 = np.zeros((Sample_number,Total_length))
    F = np.zeros((Sample_number,Total_length),dtype = complex)

    for i in range(Sample_number):
        temp_t = Samples[i]
        for j in range(Total_length):
            F1[i][j] = math.cos(temp_t*2*math.pi*j/Total_length)
            F2[i][j] = (-1)*math.sin(temp_t*2*math.pi*j/Total_length)
            F[i][j] = F1[i][j] + 1j*F2[i][j]

    M = 100

    X_axis = np.zeros(M)
    Y1_axis = np.zeros(M)
    Y2_axis = np.zeros(M)
    Y3_axis = np.zeros(M)
    for m in range(M):

        shift = -0.5 + (m+1)/M

        y = np.zeros(Sample_number,dtype = complex)
        for i in range(Sample_number):
            temp_t = Samples[i]
            p = -temp_t*2*math.pi*shift/Total_length
            y[i] = (Hadamard[i])*(math.cos(p) -1j*math.sin(p))

        x = cp.Variable(Total_length)
        constraint = [cp.norm(F @ x - y,2) <= eps]
        prob = cp.Problem(cp.Minimize(cp.norm(x,1)), constraint)
        prob.solve()

        temp_k_list = x.value
        max_idx = np.argsort(temp_k_list)[-1]
        k_solution = np.zeros(Total_length)
        k_solution[max_idx] = 1
        vec_diff = F.dot(k_solution) - y
        err = abs(np.vdot(vec_diff,vec_diff))


        X_axis[m] = shift 
        Y1_axis[m] = err
        Y2_axis[m] = (max_idx + shift)/Total_length 
        Y3_axis[m] = max_idx


    # np.save('grid_shift.npy',X_axis)
    # np.save('error.npy',Y1_axis)
    # np.save('solution.npy',Y2_axis)


    # plt.plot(X_axis,Y1_axis)
    # plt.show()

    IDX = np.argsort(Y1_axis)[0]
    output = (Y3_axis[IDX] + X_axis[IDX])/Total_length

    Output_Data[count] = abs(output - theta)


np.save('err_cs.npy',Output_Data)
np.save('total_time_cs.npy',Total_Time)