import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import qeep_prony as prony


N = 128
Original_Data = np.load('sdp_data.npy')

f1 = 0.128
f2 = 0.064
f3 = 0.032
c1 = 1/3
c2 = 1/3
c3 = 1/3
a1 = np.zeros(N,dtype = complex)
a2 = np.zeros(N,dtype = complex)
a3 = np.zeros(N,dtype = complex)
for i in range(-int(N/2),int(N/2)):
    a1[i] = (math.cos(2*math.pi*f1*(i)) + 1j*math.sin(2*math.pi*f1*(i)))
    a2[i] = (math.cos(2*math.pi*f2*(i)) + 1j*math.sin(2*math.pi*f2*(i)))
    a3[i] = (math.cos(2*math.pi*f3*(i)) + 1j*math.sin(2*math.pi*f3*(i)))

v = c1*a1 + c2*a2 + c3*a3

X2_axis = np.zeros(N)
Y2_axis = np.zeros(N)

for t in range(N):
    X2_axis[t] = t - 64
    Y2_axis[t] = Original_Data[t][N].real

M = 128
X3_axis = np.zeros(M)
Y3_axis = np.zeros(M)
Y4_axis = np.zeros(M)
for m in range(M):
    X3_axis[m] = m - 64
    Y3_axis[m] = v[m].real - Original_Data[m][N].real 
    Y4_axis[m] = v[m].imag - Original_Data[m][N].imag

# plt.scatter(X3_axis,Y3_axis)
# plt.show()

Z = np.zeros(N,dtype = complex)
for m in range(N):
    Z[m] = Original_Data[m][N] - v[m]

sigma2 = 0
for m in range(N):
    sigma2 = sigma2 + abs(Z[m])**2 

print(sigma2/N)


# HZ = prony.Hankel(Z,0,int(N/2))
# eig,vec = np.linalg.eig(HZ)
# eig_abs = np.zeros(int(N/2))
# for i in range(int(N/2)):
#     eig_abs[i] = abs(eig[i])

# print(eig_abs)