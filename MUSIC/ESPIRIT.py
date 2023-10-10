import numpy  as np
import scipy
import math
from scipy import linalg
from matplotlib import pyplot as plt
import source as srs
import qeep_prony as prony




Original_Data = np.load('sdp_data.npy')
# V = np.load('ideal_signal.npy')

L = int(math.sqrt(Original_Data.size)) - 1
dim = int(L/2)
Data = np.zeros(L,dtype = complex)
for i in range(L):
    Data[i] = Original_Data[i][L] 

# L = 50
# dim = int(L/2)
# c1 = 0.5 
# c2 = 0.5 
# k1 = 20*2*math.pi
# k2 = 21*2*math.pi
# Data = np.zeros(L,dtype = complex)
# for i in range(L):
#     val = c1*(math.cos(k1*i/L) + 1j*math.sin(k1*i/L))
#     val = val + c2*(math.cos(k2*i/L) + 1j*math.sin(k2*i/L))
#     Data[i] = val

# Data = Data + np.random.normal(0,0.05,L) + 1j*np.random.normal(0,0.05,L)
H = prony.Hankel(Data,0,dim)
S,V,D = np.linalg.svd(H) 

print(V)

M = 3
U0 = np.zeros((dim-1,M),dtype = complex)
U1 = np.zeros((dim-1,M),dtype = complex)
for l in range(dim-1):
    for m in range(M):
        U0[l][m] = S[l][m] 
        U1[l][m] = S[l+1][m]

X_axis = np.zeros(M)
Y1_axis = np.zeros(M)
Y2_axis = np.zeros(M)

PSI = (np.conj(U0).T).dot(U1)
eig,vec = np.linalg.eig(PSI)

np.save('espirit.npy',eig)

for m in range(M):
    y = eig[m].imag 
    x = eig[m].real 
    angle = math.atan(y/x)

    Y1_axis[m] = angle/(2*math.pi)

for m in range(M):
    Y1_axis[m] = eig[m].real 
    Y2_axis[m] = eig[m].imag

plt.scatter(Y1_axis,Y2_axis)
plt.show()




