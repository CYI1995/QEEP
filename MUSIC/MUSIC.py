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

H = prony.Hankel(Data,0,dim)
S,V,D = np.linalg.svd(H) 

print(V)
v1 = np.zeros(dim,dtype = complex)
v2 = np.zeros(dim,dtype = complex)
v3 = np.zeros(dim,dtype = complex)
v4 = np.zeros(dim,dtype = complex)
for l in range(dim):
    v1[l] = S[l][0]
    v2[l] = S[l][1]
    v3[l] = S[l][2]
    v4[l] = S[l][3]

P1 = prony.projector(v1,dim) + prony.projector(v2,dim) + prony.projector(v3,dim)
P2 = np.identity(dim) - P1

N = 1000
X_axis = np.zeros(N)
Y1_axis = np.zeros(N)
Y2_axis = np.zeros(N)
for n in range(N):
    w = 0.2*(n+1)/N 
    V_temp = np.zeros(dim,dtype = complex)
    for l in range(dim):
        V_temp[l] = math.cos(2*math.pi*w*(l)) + 1j*math.sin(2*math.pi*w*(l))

    V_temp = P2.dot(V_temp)

    X_axis[n] = w 
    Y1_axis[n] = math.sqrt(abs(np.vdot(V_temp,V_temp))/dim)
    Y2_axis[n] = 1/Y1_axis[n]

np.save('inverse_rw.npy',Y2_axis)
np.save('rw.npy',Y1_axis)
np.save('inverse_rw_xaxis.npy',X_axis)

plt.plot(X_axis,Y1_axis)
# plt.plot(X_axis,Y2_axis)
plt.show()
