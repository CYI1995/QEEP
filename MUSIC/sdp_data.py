import cvxpy as cp
import numpy as np
import math
from matplotlib import pyplot as plt
import source as srs


def signal(p1,p2,t):

    c1 = 0.4
    c2 = 0.6 

    f1 = 0.1
    f2 = 0.2

    val = c1*(math.cos(2*math.pi*f1*t) + 1j*math.sin(2*math.pi*f1*t))
    val = val + c2*(math.cos(2*math.pi*f2*t) + 1j*math.sin(2*math.pi*f2*t))

    return val

def Fourier_matrix(L):
    F = np.zeros((L,L),dtype = complex)
    a = 2*math.pi/L
    for i in range(L):
        for j in range(L):
            aij = a*i*j 
            F[j][i] = (math.cos(aij) - 1j*math.sin(aij))/L
            # F[j][i] = math.cos(aij)/L

    return F 

A = np.load('sdp_data.npy')

N = int(math.sqrt(A.size)) - 1
print(N)
f1 = 0.128
f2 = 0.064
f3 = 0.032
c1 = 1/3
c2 = 1/3
c3 = 1/3
a1 = np.zeros(N,dtype = complex)
a2 = np.zeros(N,dtype = complex)
a3 = np.zeros(N,dtype = complex)
for i in range(N):
    a1[i] = (math.cos(2*math.pi*f1*(i-100)) + 1j*math.sin(2*math.pi*f1*(i-100)))
    a2[i] = (math.cos(2*math.pi*f2*(i-100)) + 1j*math.sin(2*math.pi*f2*(i-100)))
    a3[i] = (math.cos(2*math.pi*f3*(i-100)) + 1j*math.sin(2*math.pi*f3*(i-100)))

v = c1*a1 + c2*a2 + c3*a3
# np.save('ideal_signal.npy',v)

X2_axis = np.zeros(N)
Y1_axis = np.zeros(N)
Y2_axis = np.zeros(N)

for t in range(N):
    X2_axis[t] = t 
    Y1_axis[t] = v[t].real
    Y2_axis[t] = A[t][N].real - v[t].real

# X1_axis = np.load('sampled_times.npy')
# L = X1_axis.size
# print(L)
# Y1_axis = np.zeros(L)
# for l in range(L):
#     idx = int(X1_axis[l])
#     Y1_axis[l] = v[idx].real

plt.scatter(X2_axis,Y2_axis,c = 'k',label = 'true_data')
# plt.scatter(X2_axis,Y1_axis,c = 'r',label = 'recovered_data')
plt.legend()
plt.show()
