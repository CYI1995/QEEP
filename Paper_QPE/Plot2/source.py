import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt

def plot_spec(M,dim):
    X = np.linspace(0,dim-1,dim)
    Y = np.zeros(dim)

    eig,vec = np.linalg.eig(M)
    eig_idx = np.argsort(eig)
    for i in range(dim):
        Y[i] = eig[eig_idx[i]].real

    plt.scatter(X,Y)
    plt.show()

def plot_phase(U,dim):
    X = np.zeros(dim)
    Y = np.zeros(dim)

    eig,vec = np.linalg.eig(U)
    for i in range(dim):
        X[i] = eig[i].real 
        Y[i] = eig[i].imag 

    plt.scatter(X,Y,marker = '.')
    # plt.show()

def dec_to_bin(num,size):
    array_temp = np.zeros(size)

    for i in range(size):
        num_temp = num%2
        array_temp[size - 1 -i] = num_temp
        num = int(num/2)

    return array_temp

def real_time_unitary(H,t):

    return scipy.linalg.expm(1j*H*t)

def adiabatic_evolution(dim,H1,H2,dt,Num):
    A = np.identity(dim,dtype = complex)

    for count in range(Num):
        s = count/Num
        H_temp = (1-s)*H1 + s*H2
        U_temp = real_time_unitary(H_temp,dt)
        A = A.dot(U_temp)

    return A

def eigenfunc(s):
    return math.sqrt(2*s*s - 2*s +1)

def projector(vec,dim):
    P = np.zeros((dim,dim),dtype = complex)
    for i in range(dim):
        for j in range(dim):
            P[i][j] = vec[i]*(np.conj(vec[j]))

    return P

def H_X(site):
    dim = 2**site
    M = np.zeros((dim,dim))

    for i in range(dim):
        bin_i = dec_to_bin(i,site)
        for j in range(site):
            if(bin_i[j] == 0):
                tar = int(i + 2**(site - 1 - j))
                M[i][tar] = 1
                M[tar][i] = 1
                
    return M

def H_Z(site):
    dim = 2**site
    M = np.zeros((dim,dim))

    for i in range(dim):
        bin_i = dec_to_bin(i,site)
        sum_of_one = 0
        for j in range(site):
            sum_of_one = sum_of_one + bin_i[j]
        M[i][i] = site - 2*sum_of_one

    return M

def H_ZZ(site):
    dim = 2**site
    M = np.zeros((dim,dim))

    for i in range(dim):
        bin_i = dec_to_bin(i,site)
        sum_of_diff = 0
        sum_of_same = 0
        for j in range(site-1):
            if(bin_i[j] != bin_i[j+1]):
                sum_of_diff = sum_of_diff + 1
            else:
                sum_of_same = sum_of_same + 1
        M[i][i] = sum_of_same - sum_of_diff

    return M

def ZZ_pair(L,site1,site2):
    dim = 2**L 
    diag = np.zeros(dim)

    for i in range(dim):
        bin_i = dec_to_bin(i,L)
        if(bin_i[site1] == bin_i[site2]):
            diag[i] = 1
        else:
            diag[i] = -1

    return np.diag(diag)

def XX_pair(L,site1,site2):
    dim = 2**L 
    M = np.zeros((dim,dim))

    for i in range(dim):
        bin_i = dec_to_bin(i,L)
        idx1 = bin_i[site1]
        idx2 = bin_i[site2]
        add = int(2**(L-1-site1)*(1-2*idx1) + 2**(L-1-site2)*(1-2*idx2))
        tar = i + add 
        M[tar][i] = 1 
        M[i][tar] = 1

    return M

def proj(vec,dim):
    P = np.zeros((dim,dim),dtype = complex)
    for i in range(dim):
        vi = vec[i]
        for j in range(dim):
            P[i][j] = np.conj(vi)*vec[j]

    return P

def matrix_norm(M,dim):

    M2 = (np.conj(M).T).dot(M)
    eig,vec = np.linalg.eig(M2)

    mx = np.argmax(eig)
    return math.sqrt(abs(eig[mx]))

def spectral_degeneracy(U,eps,dim):
    G = np.zeros(dim)
    eig,vec = np.linalg.eig(U)
    for i in range(dim):
        G[i] = abs(eig[i]-1)

    deg = 0
    for i in range(dim):
        if (G[i] < eps):
            deg = deg + 1

    return deg

def rep(H,basis,dim):
    M = np.zeros((dim,dim),dtype = complex)

    for i in range(dim):
        vec_i = basis[i]
        for j in range(dim):
            vec_j = basis[j]
            M[i][j] = np.vdot(vec_i,H.dot(vec_j))
    
    return M

def random_err():
    I = np.array([[1,0],[0,1]])
    R = np.random.normal(0, 1, 3)
    if (R[0] >0):
        O = np.array([[0,1],[1,0]])
    else:
        O = np.array([[1,0],[0,-1]])

    if(R[1] > 0):
        OO = np.kron(I,O)
    else:
        OO = np.kron(O,I)

    II = np.kron(I,I)
    if(R[2] > 0):
        return np.kron(OO,II)
    else:
        return np.kron(II,OO)

def rand_vec(dim):

    vec = np.zeros(dim)
    rand_vec = np.random.normal(0,1,dim)

    norm = math.sqrt(abs(np.vdot(rand_vec,rand_vec)))

    for i in range(dim):
        vec[i] = rand_vec[i]/norm 

    return vec

def U_evl(eig,proj,t,dim):

    U = np.zeros((dim,dim),dtype = complex)

    for i in range(dim):
        theta = eig[i].real*t 
        phase = math.cos(theta) - 1j*math.sin(theta)
        # print(phase)
        U = U + phase * proj[i] 

    return U

def norm_distance(M1,M2,dim):

    return matrix_norm(M1 - M2,dim)

def out_product(v1,v2,dim):

    M = np.zeros((dim,dim),dtype = complex)

    for i in range(dim): 
        vi = v1[i]
        for j in range(dim):
            vj = np.conj(v2[j]) 
            M[i][j] = vi*vj 

    return M