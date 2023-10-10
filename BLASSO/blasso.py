import cvxpy as cp
import numpy as np
import math
import source as srs

# def signal(p1,p2,t):

#     c1 = 0.4
#     c2 = 0.6 

#     f1 = 0.1
#     f2 = 0.5

#     val = c1*p1*(math.cos(2*math.pi*f1*t) + 1j*math.sin(2*math.pi*f1*t))
#     val = val + c2*p2*(math.cos(2*math.pi*f2*t) + 1j*math.sin(2*math.pi*f2*t))

#     return val

#X is the variable. N is the size of the signal.

N = 200
X = cp.Variable((N+1, N+1), hermitian=True)
C = np.zeros((N+1, N+1))
C[0][0] = 1 
C[N][N] = 1

f1 = 0.1
f2 = 0.5
f3 = 0.2
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
noise = np.random.normal(0,0.05,N) + 1j*np.random.normal(0,0.05,N)
# print(noise)
# T = c1*srs.out_product(a1,a1,N) + c2*srs.out_product(a2,a2,N)
v = c1*a1 + c2*a2 + c3*a3 + noise

M = 50
T = np.zeros(M)
for m in range(M):
    T[m] = np.random.randint(0,N-1)

Omega = np.zeros((N+1,N+1))
sample = np.zeros(N+1,dtype = complex)
single_column = np.zeros(N+1)
single_column[N] = 1

for m in range(M):
    t = int(T[m])
    Omega[t][t] = 1
    sample[t] = v[t]

constraints = [X >> 0]
for j in range(N-1):
    for k in range(N-j):
        constraints += [X[j+k][k] == X[j][0]]

lam = 0.05

obj = cp.Minimize(lam * cp.real(cp.trace(C @ X)) + 0.5 * cp.norm(Omega @ (X @ single_column - sample),2)**2)
prob = cp.Problem(obj, constraints)
sol = prob.solve()

np.save('sdp_data.npy',X.value)
