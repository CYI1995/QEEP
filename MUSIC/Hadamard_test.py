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


T_samples = np.load('sampled_times.npy')
print(T_samples)
Sample_number = T_samples.size
M = 200

f1 = 0.128
f2 = 0.064
f3 = 0.032

c1 = 1/3
c2 = 1/3
c3 = 1/3

N = 201
L = 100
a1 = np.zeros(N,dtype = complex)
a2 = np.zeros(N,dtype = complex)
a3 = np.zeros(N,dtype = complex)

for i in range(N):
    a1[i] = (math.cos(2*math.pi*f1*(i-L)) + 1j*math.sin(2*math.pi*f1*(i-L)))
    a2[i] = (math.cos(2*math.pi*f2*(i-L)) + 1j*math.sin(2*math.pi*f2*(i-L)))
    a3[i] = (math.cos(2*math.pi*f3*(i-L)) + 1j*math.sin(2*math.pi*f3*(i-L)))

s = c1*a1 + c2*a2 + c3*a3
r = np.arange(201)
y = np.zeros(Sample_number,dtype = complex)

Z1_sample = np.zeros(Sample_number)
Z2_sample = np.zeros(Sample_number)

for i in range(Sample_number):

    t_temp = T_samples[i]
    a = s[int(t_temp)]
    real = a.real
    imag = a.imag
    real2 = Hamadard_test(0.5*(1+real),M)
    imag2 = Hamadard_test(0.5*(1+imag),M)
    y[i] = real + 1j*imag
    # y[i] = a

np.save('Hadamard_data.npy',y)
plt.scatter(T_samples, y)
plt.plot(r,s)
plt.show()
