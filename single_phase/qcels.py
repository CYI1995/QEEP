""" Main routines for MM-QCELS 

Quantum complex exponential least squares (QCELS) can be used to
estimate the eigenvalues with reduced circuit depth. 

Last revision: 08/26/2023
"""

import scipy.io as sio
import numpy as np
import math
from copy import deepcopy
from scipy.optimize import minimize
from scipy.special import erf
import cmath
import fejer_kernel
import fourier_filter
import generate_cdf
from scipy.stats import truncnorm

def generate_QPE_distribution(spectrum,population,J):
    N = len(spectrum)
    dist = np.zeros(J)
    j_arr = 2*np.pi*np.arange(J)/J - np.pi
    for k in range(N):
        dist += population[k] * fejer_kernel.eval_Fejer_kernel(J,j_arr-spectrum[k])/J
    return dist

def generate_ts_distribution(T,NT,gamma):
    if gamma==0:
       ts=T*np.random.uniform(-1,1,NT)
    else:
       ts=truncnorm.rvs(-gamma, gamma, loc=0, scale=T, size=NT)
    return ts
    
def generate_Z_est(spectrum,population,tn,Nsample):
    N=len(tn)
    z=population.dot(np.exp(-1j*np.outer(spectrum,tn)))
    Re_true=(1+np.real(z))/2
    Im_true=(1+np.imag(z))/2
    Re_true=np.ones((Nsample, 1)) * Re_true
    Im_true=np.ones((Nsample, 1)) * Im_true
    Re_random=np.random.uniform(0,1,(Nsample,N))
    Im_random=np.random.uniform(0,1,(Nsample,N))
    Re=np.sum(Re_random<Re_true,axis=0)/Nsample
    Im=np.sum(Im_random<Im_true,axis=0)/Nsample
    Z_est = (2*Re-1)+1j*(2*Im-1)
    max_time = max(np.abs(tn))
    total_time = sum(np.abs(tn))
    return Z_est, total_time, max_time 

def generate_Z_est_multimodal(spectrum,population,T,NT,gamma):
    ts = generate_ts_distribution(T,NT,gamma)
    max_time = max(np.abs(ts))
    total_time = sum(np.abs(ts))
    Z_est, _ , _ =generate_Z_est(spectrum,population,ts,100)
    return Z_est, ts, total_time, max_time 


def generate_spectrum_population(eigenenergies, population, p):

    p = np.array(p)
    spectrum = eigenenergies * 0.25*np.pi/np.max(np.abs(eigenenergies))#normalize the spectrum
    q = population
    num_p = p.shape[0]
    q[0:num_p] = p/(1-np.sum(p))*np.sum(q[num_p:])
    return spectrum, q/np.sum(q)

def qcels_opt_fun(x, ts, Z_est):
    NT = ts.shape[0]
    N_x=int(len(x)/3)
    Z_fit = np.zeros(NT,dtype = 'complex_')
    for n in range(N_x):
       Z_fit = Z_fit + (x[3*n]+1j*x[3*n+1])*np.exp(-1j*x[3*n+2]*ts)
    return (np.linalg.norm(Z_fit-Z_est)**2/NT)

def qcels_opt_fun_coeff(x, ts, Z_est, x0):
    NT = ts.shape[0]
    N_x=int(len(x0)/3)
    Z_fit = np.zeros(NT,dtype = 'complex_')
    for n in range(N_x):
       Z_fit = Z_fit + (x[2*n]+1j*x[2*n+1])*np.exp(-1j*x0[3*n+2]*ts)
    return (np.linalg.norm(Z_fit-Z_est)**2/NT)

def qcels_opt_multimodal(ts, Z_est, x0, bounds = None, method = 'SLSQP'):
    fun = lambda x: qcels_opt_fun(x, ts, Z_est)
    N_x=int(len(x0)/3)
    bnds=np.zeros(6*N_x,dtype = 'float')
    for n in range(N_x):
       bnds[6*n]=-1
       bnds[6*n+1]=1
       bnds[6*n+2]=-1
       bnds[6*n+3]=1
       bnds[6*n+4]=-np.inf
       bnds[6*n+5]=np.inf
    bnds= [(bnds[i], bnds[i+1]) for i in range(0, len(bnds), 2)]
    if( bounds ):
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)
    else:
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)
    return res

def qcels_opt_coeff_first(ts, Z_est, x0, bounds = None, method = 'SLSQP'):
    ###need modify
    N_x=int(len(x0)/3)
    coeff=np.zeros(N_x*2)
    bnds=np.zeros(4*N_x,dtype = 'float')
    for n in range(N_x):
       bnds[4*n]=-1
       bnds[4*n+1]=1
       bnds[4*n+2]=-1
       bnds[4*n+3]=1
    bnds= [(bnds[i], bnds[i+1]) for i in range(0, len(bnds), 2)]
    for n in range(N_x):
       coeff[2*n]=x0[3*n]
       coeff[2*n+1]=x0[3*n+1]
    fun = lambda x: qcels_opt_fun_coeff(x, ts, Z_est, x0)    
    res=minimize(fun,coeff,method = 'SLSQP',bounds=bnds)
    x_out=x0
    for n in range(N_x):
       x_out[3*n]=res.x[2*n]
       x_out[3*n+1]=res.x[2*n+1]
    return x_out


def qcels_multimodal(spectrum, population, T_0, T, NT_0, NT, gamma, K, lambda_prior):        
    """Multi-level QCELS for systems with multimodal.

    Description: The code of using Multi-level QCELS to estimate the multiple dominant eigenvalues.

    Args: eigenvalues of the Hamiltonian: spectrum; 
    overlaps between the initial state and eigenvectors: population; 
    the depth for generating the data set: T_0mT; 
    number of data pairs: NT_0, NT; 
    gaussian cutoff constant: gamma; 
    initial guess of multiple dominant eigenvalues: lambda_prior
    Number of dominant eigenvalues: K
    
    Returns: an estimation of multiple dominant eigenvalues; 
    maximal evolution time T_{max}; 
    total evolution time T_{total}

    """
    total_time_all = 0.
    max_time_all = 0.
    N_level=int(np.log2(T/T_0))
    Z_est=np.zeros(NT,dtype = 'complex_')
    x0=np.zeros(3*K,dtype = 'float')
    Z_est, ts, total_time, max_time=generate_Z_est_multimodal(
        spectrum,population,T_0,NT_0,gamma) #Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
    total_time_all += total_time
    max_time_all = max(max_time_all, max_time)
    N_initial=10
    lambda_prior_collect=np.zeros((N_initial,len(lambda_prior)),dtype = 'float')
    lambda_prior_collect[0,:]=lambda_prior
    for n in range(N_initial-1):
        lambda_prior_collect[n+1,:]=np.random.uniform(spectrum[0],spectrum[-1],K)
    #Step up and solve the optimization problem
    Residue=np.inf
    for p in range(N_initial):#try different initial to make sure find global minimal
        lambda_prior_new=lambda_prior_collect[p,:]
        for n in range(K):
           x0[3*n:3*n+3]=np.array((np.random.uniform(0,1),0,lambda_prior_new[n]))
        x0 = qcels_opt_coeff_first(ts, Z_est, x0)
        res = qcels_opt_multimodal(ts, Z_est, x0)#Solve the optimization problem
        if res.fun<Residue:
            x0_fix=np.array(res.x)
            Residue=res.fun
    #Update initial guess for next iteration
    #Update the estimation interval
    x0=x0_fix
    bnds=np.zeros(6*K,dtype = 'float')
    for n in range(K):
       bnds[6*n]=-np.infty
       bnds[6*n+1]=np.infty
       bnds[6*n+2]=-np.infty
       bnds[6*n+3]=np.infty
       bnds[6*n+4]=x0[3*n+2]-np.pi/T_0
       bnds[6*n+5]=x0[3*n+2]+np.pi/T_0
    bnds= [(bnds[i], bnds[i+1]) for i in range(0, len(bnds), 2)]
    #Iteration
    for n_QCELS in range(N_level):
        T=T_0*2**(n_QCELS+1)
        Z_est, ts, total_time, max_time=generate_Z_est_multimodal(
            spectrum,population,T,NT,gamma) #Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
        total_time_all += total_time
        max_time_all = max(max_time_all, max_time)
        #Step up and solve the optimization problem
        res = qcels_opt_multimodal(ts, Z_est, x0, bounds=bnds)#Solve the optimization problem
        #Update initial guess for next iteration
        x0=np.array(res.x)
        #Update the estimation interval
        bnds=np.zeros(6*K,dtype = 'float')
        for n in range(K):
           bnds[6*n]=-np.infty
           bnds[6*n+1]=np.infty
           bnds[6*n+2]=-np.infty
           bnds[6*n+3]=np.infty
           bnds[6*n+4]=x0[3*n+2]-np.pi/T
           bnds[6*n+5]=x0[3*n+2]+np.pi/T
        bnds= [(bnds[i], bnds[i+1]) for i in range(0, len(bnds), 2)]
    #print(x0,'one iteration ends',T)
    return x0, total_time_all, max_time_all

if __name__ == "__main__":
    import scipy.io as sio
    import numpy as np
    from copy import deepcopy
    from scipy.optimize import minimize
    from matplotlib import pyplot as plt
    from scipy.special import erf
    from mpl_toolkits.mplot3d import Axes3D
    import cmath
    import matplotlib
    import fejer_kernel
    import fourier_filter
    import generate_cdf
    matplotlib.rcParams['font.size'] = 15
    matplotlib.rcParams['lines.markersize'] = 10



theta = math.sqrt(math.pi)/(2*math.pi)
spectrum = np.array([theta,0.5])
population = np.array([0.9,0.1])
T_0 = 200
T = T_0
NT_0 = 40000
NT = 40000
gamma = 1
K = 2
lambda_prior = np.array([0.28,0.5])


Batch = 100
Output_Data = np.zeros(Batch)
Total_Time = np.zeros(Batch)

for count in range(Batch):
    X, T_1, M = qcels_multimodal(spectrum, population, T_0, T, NT_0, NT, gamma, K, lambda_prior)
    output = X[2]

    Output_Data[count] = abs(output - theta)
    Total_Time[count] = T_1

np.save('err_qcels.npy',Output_Data)
np.save('total_time_qcels.npy',Total_Time)

# for n in range(K):

#     print("population = ",X[3*n]+1j*X[3*n+1])
#     print("spectrum = ",X[3*n+2])

# print(T)
