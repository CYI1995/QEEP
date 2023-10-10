import cvxpy as cp
import numpy as np
import math
from matplotlib import pyplot as plt
import random

def Dirichlet(N,k):

    val = 0

    for n in range(-N,N+1):

        val = val + math.cos(n*math.pi*k) + 1j*math.sin(n*math.pi*k)

    return val

# def Dirichlet_derivative(N,k):

#     val = 0

#     for n in range(-N,N+1):

#         val = val + (n*math.pi)*( - math.sin(n*math.pi*k))

#     return val

def Dirichlet_derivative(N,k):

    val = 0

    for n in range(1,N+1):

        val = val - 2*math.pi*n*math.sin(n*math.pi*k)

    return val


N = 200

L = 1000

X = np.zeros(L)
Y = np.zeros(L)
Z = np.zeros(L)

a = 0.5/N 
r = 1/(8*math.sqrt(6))

for l in range(L):

    k = a*(-1 + 2*(l+1)/L)

    y = Dirichlet(N,k) - 0.1*Dirichlet(N,k - 0.5*a)

    X[l] = k
    Y[l] = y.real



# print(r)

# for l in range(L):

#     N = 80 + 20*(l+1)
#     X[l] = N
#     Y[l] = abs(Dirichlet_derivative(N,1.4/N))
#     # Z[l] = abs(Dirichlet_derivative(N,1.5/N))
#     Z[l] = 0.1*N**2


plt.plot(X,Y)
# plt.plot(X,Z,label = 'k = 0.5')
# plt.loglog()
# plt.legend()
plt.show() 