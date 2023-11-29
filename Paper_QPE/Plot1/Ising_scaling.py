import cvxpy as cp
import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
import random
import source as srs 
import statistics

def Hamadard_test(p,M):

    val = 0
    for m in range(M):
        r = random.uniform(0,1)
        if(r < p):
            val = val + 1
        else:
            val = val - 1

    return val/M

def signal(frequencies,amplitudes,t):

    L = len(frequencies)

    val = 0 + 1j*0
    for i in range(L):
        k = frequencies[i]
        val = val + amplitudes[i]*(math.cos(k*t) - 1j*math.sin(k*t))
    
    return val

def func_variance(A):

    x2_mean = 0

    for i in range(len(A)):
        x2_mean = x2_mean + A[i]**2 

    return math.sqrt((x2_mean/len(A) - (np.sum(A)/len(A))**2))

frequencies = np.load('frequencies.npy')
amplitudes = np.load('amplitudes.npy')

print(amplitudes[0])

idx = np.argsort(frequencies)[-1]
gs_energy = frequencies[idx]

Batch = 50
Ns = 100

Variance = []
Average_err = []
Total_time_set = []
Maximal_time_set = []

N_T_total = 10
for sth in range(N_T_total):

    Total_length = 11 + 50*(sth+1)
    Sample_number = int(2*math.log(Total_length))
    print('Total length = ', Total_length,', Sample number = ', Sample_number)

    Output_Data = np.zeros(Batch)
    Total_Time = np.zeros(Batch)
    Max_Time = np.zeros(Batch)

    for count in range(Batch):

        Samples = np.zeros(Sample_number)
        for i in range(Sample_number):
            temp_t = np.random.randint(0,Total_length-1)
            Samples[i] = temp_t

        idx = np.argmax(Samples)
        T_m = Samples[idx]
        Max_Time[count] = T_m
        Hadamard = np.zeros(Sample_number,dtype = complex)

        for i in range(Sample_number):
            t_temp = Samples[i]
            signal_temp = signal(frequencies,amplitudes,t_temp)
            real = signal_temp.real 
            imag = signal_temp.imag
            # Hadamard[i] = real + 1j*imag
            real2 = Hamadard_test(0.5*(1+real),Ns)
            imag2 = Hamadard_test(0.5*(1+imag),Ns)
            Hadamard[i] = real2 + 1j*imag2

        T_tol = Ns*np.sum(Samples)
        Total_Time[count] = T_tol

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

        eps = math.sqrt(Sample_number)*(2*math.pi/M +  math.sqrt(Sample_number/Ns))
        # eps = 3*math.sqrt(Sample_number/Ns)

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

            X_axis[m] = shift 
            Y1_axis[m] = abs(np.vdot(vec_diff,vec_diff))
            Y3_axis[m] = max_idx

        IDX = np.argsort(Y1_axis)[0]
        output = (Y3_axis[IDX] + X_axis[IDX])/Total_length

        err = min(abs(2*math.pi*(output-1) - gs_energy),abs(2*math.pi*(output) - gs_energy))
        Output_Data[count] = err

    Variance.append(func_variance(Output_Data))
    Average_err.append(np.sum(Output_Data)/Batch)
    Total_time_set.append(np.sum(Total_Time)/Batch)
    Maximal_time_set.append(np.sum(Max_Time)/Batch)

np.save('Variance_err_cs_g1.npy',Variance)
np.save('Average_err_cs_g1.npy',Average_err)
np.save('Average_Total_time_cs_g1.npy',Total_time_set)
np.save('Average_Maximal_time_cs_g1.npy',Maximal_time_set)