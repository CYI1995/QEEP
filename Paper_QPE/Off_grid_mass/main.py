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

    std_A = 0
    for i in range(len(A)):
        std_A = std_A + A[i]**2

    return math.sqrt(std_A/(len(A)-1))

def one_norm(vec,dim):

    norm = 0
    for i in range(dim):
        norm = norm + abs(vec[i])

    return norm

def GenerateRandomTimeSamples(Sample_ratio,Total_length):

    Omega = np.zeros(Total_length)
    for i in range(Total_length):
        random_num_temp = random.uniform(0,1)
        if(random_num_temp < Sample_ratio[i]):
            Omega[i] = 1

    return Omega

def GenerateSignalSample(Sample_number,frequencies,amplitudes,Ns):

    True_signal = np.zeros(Sample_number, dtype = complex)
    Hadamard = np.zeros(Sample_number,dtype = complex)
    for i in range(Sample_number):
        t_temp = i
        signal_temp = signal(frequencies,amplitudes,t_temp)
        real = signal_temp.real 
        imag = signal_temp.imag
        True_signal[i] = real + 1j*imag
        real2 = Hamadard_test(0.5*(1+real),Ns)
        imag2 = Hamadard_test(0.5*(1+imag),Ns)
        Hadamard[i] = real2 + 1j*imag2

    return True_signal,Hadamard

def BasisPursuitDenoising(y_omega,F_omega,eps,grid_shift,Total_length):

    shifted_y_omega = np.zeros(Total_length,dtype = complex)
    for i in range(Total_length):
        p = -i*2*math.pi*grid_shift/Total_length
        shifted_y_omega[i] = (y_omega[i])*(math.cos(p) -1j*math.sin(p))

    x = cp.Variable(Total_length)
    constraint = [cp.norm(F_omega @ x - shifted_y_omega,2) <= eps]
    prob = cp.Problem(cp.Minimize(cp.norm(x,1)), constraint)
    prob.solve()

    return prob.status, x.value

def BasisPursuit(Omega,y_omega,F_omega,grid_shift,Sample_number,Total_length):

    y = np.zeros(Sample_number,dtype = complex)
    for i in range(Sample_number):
        temp_t = Omega[i]
        p = -temp_t*2*math.pi*grid_shift/Total_length
        y[i] = (y_omega[i])*(math.cos(p) -1j*math.sin(p))

    x = cp.Variable(Total_length)
    constraint = [F_omega @ x == y]
    prob = cp.Problem(cp.Minimize(cp.norm(x,1)), constraint)
    prob.solve()

    return prob.status, x.value

def Virtual_Fourier_inverse(v_omega,Omega,Sample_number,Total_length):

    v = np.zeros(Total_length,dtype = complex)
    for n in range(Total_length):
        entry = 0 + 0*1j
        for t in range(Sample_number):
            entry = entry + math.cos(2*math.pi*n*Omega[t]/Total_length) + 1j*math.sin(2*math.pi*n*Omega[t]/Total_length)
        v[n] = entry/Total_length

def Fourier(v,N):

    F = np.zeros((N,N),dtype = complex)
    for i in range(N):
        t_temp = i
        for j in range(N):
            f_temp = 2*math.pi*(j-v)/N
            F[i][j] = math.cos(f_temp*t_temp) + (-1j)*math.sin(f_temp*t_temp)
    return F

def Dominant_frequency_erstimation(y_Omega,Total_length,Search_Parameter):

    Cost = np.zeros(Searching_Parameter)
    for n in range(Searching_Parameter):
        k_temp = (n+1)/Searching_Parameter
        Cost[n] = abs(np.vdot(y_Omega,atom(k_temp,Total_length)))

    k_match = np.argmax(Cost)/Searching_Parameter
    
    return k_match

def off_grid_mass(y):

    L= 200

    N = len(y)
    F = Fourier(0,N)
    Mass_list = np.zeros(L)
    signal_temp = np.zeros(N,dtype = complex)

    for l in range(L):
        v = -0.5 + (l+0.5)/L
        for t in range(N):
            signal_temp[t] = y[t]*(math.cos(2*math.pi*v*t/N) + 1j*math.sin(2*math.pi*v*t/N)) 
        x_imag = (F.dot(signal_temp)).imag
        y_imag = (np.conj(F).T).dot(x_imag)
        Mass_list[l] = np.vdot(y_imag,y_imag).real
    
    idx_opt = np.argmin(Mass_list)
    v_opt = -0.5 + (idx_opt + 0.5)/L
    signal_opt = np.zeros(N,dtype = complex)
    for t in range(N):
        signal_opt[t] = y[t]*(math.cos(2*math.pi*v_opt*t/N) + 1j*math.sin(2*math.pi*v_opt*t/N)) 
    x_imag_opt = (F.dot(signal_opt)).imag
    y_imag_opt = (np.conj(F).T).dot(x_imag_opt)/N

    return math.sqrt(abs(np.vdot(y_imag_opt,y_imag_opt))/N)



data1 = np.load('frequencies_ising.npy')
frequencies_ising = np.zeros(10)
for i in range(10):
    frequencies_ising[i] = data1[i]


Number_of_Hadamard_tests = 100
N_T_total = 5
A = np.array([1/8,1/4,1/2])
Hadamard_err_list = np.zeros((3,N_T_total))
Off_gird_mass_list = np.zeros((3,N_T_total))

for set in range(3):

    alpha = A[set]
    amplitudes = np.zeros(10)
    for i in range(10):
        amplitudes[i] = alpha**i 
    amplitudes[i] = amplitudes[i]/np.sum(amplitudes)


    for n_t in range(N_T_total):

        Total_length = int(100*1.4**(n_t+1))
        F = Fourier(0,Total_length)

        y0,y = GenerateSignalSample(Total_length,frequencies_ising,amplitudes,Number_of_Hadamard_tests)

        Hadamard_err = math.sqrt(np.vdot(y - y0, y - y0).real/Total_length)
        Minima_off_grid = off_grid_mass(y0)

        Hadamard_err_list[set][n_t] = Hadamard_err
        Off_gird_mass_list[set][n_t] = Minima_off_grid 

np.save('Hadamard_err_ising.npy',Hadamard_err_list)
np.save('Off_grid_mass_ising.npy',Off_gird_mass_list)

        
data2 = np.load('frequencies_hubbard.npy')
frequencies_hubbard = np.zeros(10)
for i in range(10):
    frequencies_hubbard[i] = data2[i]


Number_of_Hadamard_tests = 100
N_T_total = 5
A = np.array([1/8,1/4,1/2])
Hadamard_err_list = np.zeros((3,N_T_total))
Off_gird_mass_list = np.zeros((3,N_T_total))

for set in range(3):

    alpha = A[set]
    amplitudes = np.zeros(10)
    for i in range(10):
        amplitudes[i] = alpha**i 
    amplitudes[i] = amplitudes[i]/np.sum(amplitudes)


    for n_t in range(N_T_total):

        Total_length = int(100*1.4**(n_t+1))
        F = Fourier(0,Total_length)

        y0,y = GenerateSignalSample(Total_length,frequencies_hubbard,amplitudes,Number_of_Hadamard_tests)

        Hadamard_err = math.sqrt(np.vdot(y - y0, y - y0).real/Total_length)
        Minima_off_grid = off_grid_mass(y0)

        Hadamard_err_list[set][n_t] = Hadamard_err
        Off_gird_mass_list[set][n_t] = Minima_off_grid 

np.save('Hadamard_err_hubbard.npy',Hadamard_err_list)
np.save('Off_grid_mass_hubbard.npy',Off_gird_mass_list)