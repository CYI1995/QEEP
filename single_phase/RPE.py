import cvxpy as cp
import numpy as np
import math
from matplotlib import pyplot as plt
import random

def Hamadard_test(p,M):

    val = 0
    for m in range(M):
        r = random.uniform(0,1)
        if(r < p):
            val = val + 1
        else:
            val = val - 1

    return val/M

# def random_noise(t):

#     k1 = 0.25
#     k2 = 0.5

#     # N = np.random.normal(0,0.1,2)

#     p1 = math.cos(2*math.pi*k1*t) + 1j*math.sin(2*math.pi*k1*t)
#     p2 = math.cos(2*math.pi*k2*t) + 1j*math.sin(2*math.pi*k2*t)

#     signal = 0.9*p1 + 0.1*p2

#     # real = signal.real
#     # imag = signal.imag
#     # real2 = Hamadard_test(0.5*(1+real),100)
#     # imag2 = Hamadard_test(0.5*(1+imag),100)

#     # return 0.9*p1 + 0.1*p2 + N[0] + 1j*N[1]
#     return signal
#     # return real2 + 1j*imag2

def random_noise(t):

    k1 = 1/6
    k2 = 1/3
    p1 = math.cos(2*math.pi*k1*t) + 1j*math.sin(2*math.pi*k1*t)
    p2 = math.cos(2*math.pi*k2*t) + 1j*math.sin(2*math.pi*k2*t)

    x = 0.000
    return (1-x)*p1 + x*p2

def arctan(cos,sin):

    a1 = math.acos(cos)

    if(sin <= 0):
        a1 = 2*math.pi - a1 

    return a1

def mod2pi(a):
    
    if(a<0):
        a = a + 2*math.pi 
    return a

def find_closest(Array, L, Tar):

    idx = 0
    d0 = abs(Array[0] - Tar)
    for i in range(1,L):
        d_temp = abs(Array[i] - Tar)
        if(d_temp < d0):
            idx = i 
            d0 = d_temp 

    return Array[idx]

Ns = 100

# X_axis = np.zeros(J)
# Y_axis = np.zeros(J)
# Z_axis = np.zeros(J)
theta_dec = 0
f1 = math.sqrt(math.pi)/(2*math.pi)
f2 = 0.5
Total_time = 0

Batch = 100
Output_Data = np.zeros(Batch)
Total_Time = np.zeros(Batch)

for count in range(Batch):

    J = 11
    T_tot = 0

    for j in range(J):

        t = 2**j 
        T_tot = T_tot + t*Ns

        s1 = math.cos(2*math.pi*f1*t) + 1j*math.sin(2*math.pi*f1*t)
        s2 = math.cos(2*math.pi*f2*t) + 1j*math.sin(2*math.pi*f2*t)
        signal = 0.9*s1 + 0.1*s2

        Re = signal.real
        Im = signal.imag 

        signal = Hamadard_test(Re,Ns) + 1j*Hamadard_test(Im,Ns) 

        aj = np.arctan2(signal.imag,signal.real)
        if(signal.imag <= 0):
            aj = aj + 2*math.pi

        aj_dec = aj/(2*math.pi)

        Sj = np.zeros(t)
        for i in range(t):
            Sj[i] = ((aj_dec + i)%t)/t

        theta_dec_new = find_closest(Sj,t,theta_dec)

        # X_axis[j] = j 
        # Y_axis[j] = abs(theta_dec_new - f1)
        # Z_axis[j] = (theta_dec_new - theta_dec)%1

        theta_dec = theta_dec_new

# print(abs(theta_dec - f1))

    Output_Data[count] = abs(theta_dec - f1) 
    Total_Time[count] = T_tot

np.save('err_rpe.npy',Output_Data)
np.save('total_time_rpe.npy',Total_Time)




