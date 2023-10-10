import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import string

def kernel_f(k,Y,N):

    val = 0 
    L = int(N/2)

    for n in range(1,N):
        t = n - L
        val = val + Y[n]*(math.cos(2*math.pi*k*t) - 1j*math.sin(2*math.pi*k*t))

    return val



Original_Data = np.load('sdp_data.npy')

N = 128
f1 = 0.128
f2 = 0.064
f3 = 0.052
c1 = 1/3
c2 = 1/3
c3 = 1/3

X2_axis = np.zeros(N)
Y2_axis = np.zeros(N,dtype = complex)

for t in range(N):
    X2_axis[t] = t - int(N/2)
    Y2_axis[t] = Original_Data[t][N]

X3_axis = np.zeros(N)
Y3_axis = np.zeros(N,dtype = complex)
for n in range(N):
    t = n - int(N/2)
    val = 0
    val = val + c1*(math.cos(2*math.pi*f1*t)) + 1j*c1*(math.sin(2*math.pi*f1*t))
    val = val + c2*(math.cos(2*math.pi*f2*t)) + 1j*c2*(math.sin(2*math.pi*f2*t))
    val = val + c3*(math.cos(2*math.pi*f3*t)) + 1j*c3*(math.sin(2*math.pi*f3*t))
    X3_axis[n] = t 
    Y3_axis[n] = val

M = 1000 
X1_axis = np.zeros(M)
Y1_axis = np.zeros(M)
Z1_axis = np.zeros(M)
for m in range(M):

    k = (m+1)/M 
    val = kernel_f(k,Y2_axis,N)
    Y1_axis[m] = val.real 
    Z1_axis[m] = val.imag 
    X1_axis[m] = k

plt.plot(X2_axis,Y2_axis)
# plt.plot(X1_axis,Z1_axis)
plt.show()
