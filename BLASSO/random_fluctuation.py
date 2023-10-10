import cvxpy as cp
import numpy as np
import math
from matplotlib import pyplot as plt
import source as srs


def Fejer(j,M):

    val = 0
    if(j > 0):
        k1 = j-M
        k2 = M 
    else:
        k1 = -M 
        k2 = j+M 


    for k in range(k1,k2+1):
        val = val + (1 - abs(k/M))*(1 - abs((j-k)/M))

    return val/M

def kernel(f,g,L):

    val = 0 
    M = int((L-1)/2)
    for i in range(L):
        a = 2*math.pi*f*(i+2*M)
        val = val + g[i]*(math.cos(a) - 1j*math.sin(a))

    return val/M

def random_kernel(f,g,L,T):

    val = 0 
    M = int((L-1)/4)
    size = T.size
    for i in range(size):
        idx = int(T[i])
        a = 2*math.pi*f*(idx-2*M)
        val = val + g[int(idx)]*(math.cos(a) - 1j*math.sin(a))

    return val/M

def random_kernel_d(f,g,L,T):

    val = 0 
    M = int((L-1)/4)
    size = T.size
    for i in range(size):
        idx = int(T[i])
        a = 2*math.pi*f*(idx-2*M)
        val = val + g[int(idx)]*(math.cos(a) - 1j*math.sin(a))*(-1j*2*math.pi*(idx - 2*M))

    return val/M

def random_kernel_dd(f,g,L,T):

    val = 0 
    M = int((L-1)/4)
    size = T.size
    for i in range(size):
        idx = int(T[i])
        a = 2*math.pi*f*(idx-2*M)
        val = val + g[int(idx)]*(math.cos(a) - 1j*math.sin(a))*(-1j*2*math.pi*(idx - 2*M))**2

    return val/M

T = np.load('sampled.npy')
print(T)
v = 0.25 
M = 100
L = int(4*M+1)

S = 4 
U = np.zeros(int(2*S))
for i in range(S):
    U[i] = 1 
F = np.array([0.25,0.5,0.75,1])

g = np.zeros(L)
for j in range(L):
    js = j - 2*M
    g[j] = Fejer(js,M)


D = np.zeros((int(2*S),int(2*S)),dtype = complex)
for i in range(S):
    f_i = F[i]
    for j in range(S):
        f_j = F[j]
        D[i][j] = random_kernel(f_i - f_j, g, L,T)
        D[i+S][j] = random_kernel_d(f_i - f_j, g, L,T)
        D[i][j+S] = random_kernel_d(f_i - f_j, g, L,T) 
        D[i+S][j+S] = random_kernel_dd(f_i - f_j, g, L,T) 

# srs.plot_spec(D,int(2*S))
# plt.show()

A = np.linalg.inv(D).dot(U)

# print(A)

N = 100 
X_axis = np.zeros(N)
Y1_axis = np.zeros(N)
Y2_axis = np.zeros(N)
for i in range(N):
    f = (i+1)/N 
    y = 0 + 1j*0
    for j in range(S):
        a_j = A[j]
        b_j = A[j+S]
        f_j = F[j]
        y = y + a_j*random_kernel(f-f_j,g,L,T)
        y = y + b_j*random_kernel_d(f-f_j,g,L,T)
        
    X_axis[i] = f 
    Y1_axis[i] = y.real 
    Y2_axis[i] = abs(y)

plt.plot(X_axis,Y1_axis)
# plt.plot(X_axis,Y2_axis)
plt.show()
