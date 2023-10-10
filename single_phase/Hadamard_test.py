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



Sample_number = 50
Total_length = 1000 

k1 = 20.25
k2 = 100 
c1 = 0.9 
c2 = 0.1

T_samples = np.load('sampled_times.npy')
M = 100

y = np.zeros(Sample_number,dtype = complex)
for i in range(Sample_number):

    t_temp = T_samples[i]
    a1 = 2*math.pi*t_temp*k1/Total_length
    a2 = 2*math.pi*t_temp*k2/Total_length
    real = c1*math.cos(a1) + c2*math.cos(a2)
    imag = c1*math.sin(a1) + c2*math.sin(a2)
    real2 = Hamadard_test(0.5*(1+real),M)
    imag2 = Hamadard_test(0.5*(1+imag),M)
    y[i] = real2 - 1j*imag2

    print(abs(real2 - real + 1j*imag2 - 1j*imag))


np.save('Hadamard_data.npy',y)
    