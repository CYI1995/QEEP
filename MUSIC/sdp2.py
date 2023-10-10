import cvxpy as cp
import numpy as np
import math
import source as srs

def signal(p1,p2,t):

    c1 = 0.4
    c2 = 0.6 

    f1 = 0.1
    f2 = 0.2

    val = c1*p1*(math.cos(2*math.pi*f1*t) + 1j*math.sin(2*math.pi*f1*t))
    val = val + c2*p2*(math.cos(2*math.pi*f2*t) + 1j*math.sin(2*math.pi*f2*t))

    return val



#X is the variable. N is the size of the signal.
N = 200
X = cp.Variable((N+1, N+1), hermitian=True)
C = np.zeros((N+1, N+1))
C[0][0] = 1 
C[N][N] = 1

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
    a1[i] = (math.cos(2*math.pi*f1*(i-L)) + 1j*math.sin(2*math.pi*f1*(i-L)))
    a2[i] = (math.cos(2*math.pi*f2*(i-L)) + 1j*math.sin(2*math.pi*f2*(i-L)))
    a3[i] = (math.cos(2*math.pi*f3*(i-L)) + 1j*math.sin(2*math.pi*f3*(i-L)))
# noise = np.random.normal(0,0.05,N) + 1j*np.random.normal(0,0.05,N)
# print(noise)
T = c1*srs.out_product(a1,a1,N) + c2*srs.out_product(a2,a2,N)
v = c1*a1 + c2*a2 + c3*a3

M = 20
T = np.zeros(M)
for m in range(M):
    T[m] = np.random.randint(0,N-1)

constraints = [X >> 0]
for m in range(M):
    t = int(T[m])
    constraints += [cp.abs(X[N][t] - np.conj(v[t])) <= 0.01]

for j in range(N-1):
    for k in range(N-j):
        constraints += [X[j+k][k] == X[j][0]]

obj = cp.Minimize(cp.real(cp.trace(C @ X)))
prob = cp.Problem(obj, constraints)
sol = prob.solve()

np.save('sdp_data.npy',X.value)

