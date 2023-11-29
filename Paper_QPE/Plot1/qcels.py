""" Main routines for QCELS 

Quantum complex exponential least squares (QCELS) can be used to
estimate the ground-state energy with reduced circuit depth. 

Last revision: 11/22/2022
"""

import scipy.io as sio
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from scipy.special import erf
import statistics

def generate_QPE_distribution(spectrum,population,J):
    N = len(spectrum)
    dist = np.zeros(J)
    j_arr = 2*np.pi*np.arange(J)/J - np.pi
    for k in range(N):
        dist += population[k] * fejer_kernel.eval_Fejer_kernel(J,j_arr-spectrum[k])/J
    return dist

def get_estimated_ground_energy_rough(d,delta,spectrum,population,Nsample,Nbatch):
    
    F_coeffs = fourier_filter.F_fourier_coeffs(d,delta)

    compute_prob_X = lambda T: generate_cdf.compute_prob_X_(T,spectrum,population)
    compute_prob_Y = lambda T: generate_cdf.compute_prob_Y_(T,spectrum,population)


    outcome_X_arr, outcome_Y_arr, J_arr = generate_cdf.sample_XY(compute_prob_X, 
                                compute_prob_Y, F_coeffs, Nsample, Nbatch) #Generate sample to calculate CDF

    total_evolution_time = np.sum(np.abs(J_arr))
    average_evolution_time = total_evolution_time/(Nsample*Nbatch)
    maxi_evolution_time=max(np.abs(J_arr[0,:]))

    Nx = 10
    Lx = np.pi/3
    ground_energy_estimate = 0.0
    count = 0
    #---"binary" search
    while Lx > delta:
        x = (2*np.arange(Nx)/Nx-1)*Lx +  ground_energy_estimate
        y_avg = generate_cdf.compute_cdf_from_XY(x, outcome_X_arr, outcome_Y_arr, J_arr, F_coeffs)#Calculate the value of CDF
        indicator_list = y_avg > (population[0]/2.05)
        ix = np.nonzero(indicator_list)[0][0]
        ground_energy_estimate = x[ix]
        Lx = Lx/2
        count += 1
    
    return ground_energy_estimate, count*total_evolution_time, maxi_evolution_time

def generate_filtered_Z_est(spectrum,population,t,x,d,delta,Nsample,Nbatch):
    
    F_coeffs = fourier_filter.F_fourier_coeffs(d,delta)
    compute_prob_X = lambda t_: generate_cdf.compute_prob_X_(t_,spectrum,population)
    compute_prob_Y = lambda t_: generate_cdf.compute_prob_Y_(t_,spectrum,population)
    #Calculate <\psi|F(H)\exp(-itH)|\psi>
    outcome_X_arr, outcome_Y_arr, J_arr = generate_cdf.sample_XY_QCELS(compute_prob_X, 
                                compute_prob_Y, F_coeffs, Nsample, Nbatch,t) #Generate samples using Hadmard test
    y_avg = generate_cdf.compute_cdf_from_XY_QCELS(x, outcome_X_arr, outcome_Y_arr, J_arr, F_coeffs) 
    total_time = np.sum(np.abs(J_arr))+t*Nsample*Nbatch
    max_time= max(np.abs(J_arr[0,:]))+t
    return y_avg, total_time, max_time


def generate_Z_est(spectrum,population,t,Nsample):
    Re=0
    Im=0
    z=np.dot(population,np.exp(-1j*spectrum*t))
    Re_true=(1+z.real)/2
    Im_true=(1+z.imag)/2
    #Simulate Hadmard test
    for nt in range(Nsample):
        if np.random.uniform(0,1)<Re_true:
           Re+=1
    for nt2 in range(Nsample):
        if np.random.uniform(0,1)<Im_true:
           Im+=1
    Z_est = complex(2*Re/Nsample-1,2*Im/Nsample-1)
    max_time = t
    total_time = t * Nsample
    return Z_est, total_time, max_time 
       

def generate_spectrum_population(eigenenergies, population, p):

    p = np.array(p)
    spectrum = eigenenergies * 0.25*np.pi/np.max(np.abs(eigenenergies))#normalize the spectrum
    q = population
    num_p = p.shape[0]
    q[0:num_p] = p/(1-np.sum(p))*np.sum(q[num_p:])
    return spectrum, q/np.sum(q)

def qcels_opt_fun(x, ts, Z_est):
    NT = ts.shape[0]
    Z_fit=np.zeros(NT,dtype = 'complex_')
    Z_fit=(x[0]+1j*x[1])*np.exp(-1j*x[2]*ts)
    return (np.linalg.norm(Z_fit-Z_est)**2/NT)

def qcels_opt(ts, Z_est, x0, bounds = None, method = 'SLSQP'):

    fun = lambda x: qcels_opt_fun(x, ts, Z_est)
    if( bounds ):
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)
    else:
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)

    return res


def qcels_largeoverlap(spectrum, population, T, NT, Nsample, lambda_prior):
    """Multi-level QCELS for a system with a large initial overlap.

    Description: The code of using Multi-level QCELS to estimate the ground state energy for a systems with a large initial overlap

    Args: eigenvalues of the Hamiltonian: spectrum; 
    overlaps between the initial state and eigenvectors: population; 
    the depth for generating the data set: T; 
    number of data pairs: NT; 
    number of samples: Nsample; 
    initial guess of \lambda_0: lambda_prior

    Returns: an estimation of \lambda_0; 
    maximal evolution time T_{max}; 
    total evolution time T_{total}

    """
    total_time_all = 0.
    max_time_all = 0.

    N_level=int(np.log2(T/NT))
    Z_est=np.zeros(NT,dtype = 'complex_')
    tau=T/NT/(2**N_level)
    ts=tau*np.arange(NT)
    for i in range(NT):
        Z_est[i], total_time, max_time=generate_Z_est(
                spectrum,population,ts[i],Nsample) #Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
        # print(total_time)
        total_time_all += total_time
        max_time_all = max(max_time_all, max_time)
    #Step up and solve the optimization problem
    x0=np.array((0.5,0,lambda_prior))
    res = qcels_opt(ts, Z_est, x0)#Solve the optimization problem
    #Update initial guess for next iteration
    ground_coefficient_QCELS=res.x[0]
    ground_coefficient_QCELS2=res.x[1]
    ground_energy_estimate_QCELS=res.x[2]
    #Update the estimation interval
    lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau) 
    lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau) 
    #Iteration
    for n_QCELS in range(N_level):
        Z_est=np.zeros(NT,dtype = 'complex_')
        tau=T/NT/(2**(N_level-n_QCELS-1)) #generate a sequence of \tau_j
        ts=tau*np.arange(NT)
        for i in range(NT):
            Z_est[i], total_time, max_time=generate_Z_est(
                    spectrum,population,ts[i],Nsample) #Approximate <\psi|\exp(-itH)|\psi> using Hadmard test
            total_time_all += total_time
            max_time_all = max(max_time_all, max_time)
        #Step up and solve the optimization problem
        x0=np.array((ground_coefficient_QCELS,ground_coefficient_QCELS2,ground_energy_estimate_QCELS))
        bnds=((-np.inf,np.inf),(-np.inf,np.inf),(lambda_min,lambda_max)) 
        res = qcels_opt(ts, Z_est, x0, bounds=bnds)#Solve the optimization problem
        #Update initial guess for next iteration
        ground_coefficient_QCELS=res.x[0]
        ground_coefficient_QCELS2=res.x[1]
        ground_energy_estimate_QCELS=res.x[2]
        #Update the estimation interval
        lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau) 
        lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau) 

    return res, total_time_all, max_time_all

def func_variance(A):

    x2_mean = 0

    for i in range(len(A)):
        x2_mean = x2_mean + A[i]**2 

    return cmath.sqrt((x2_mean/len(A) - (np.sum(A)/len(A))**2)).real

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
    import source as srs
    import statistics
    matplotlib.rcParams['font.size'] = 15
    matplotlib.rcParams['lines.markersize'] = 10

    num_sites = 8
    dim = 2**num_sites
    J = 1.0
    g = 4
    
    num_eigenstates_max = 100
    

    ham = -J*srs.H_ZZ(num_sites) - J*srs.ZZ_pair(num_sites,0,num_sites-1) + g*srs.H_X(num_sites)
    eig,vec = np.linalg.eig(ham)

    spectrum = np.load('frequencies.npy')
    population = np.load('amplitudes.npy')

    p0_array = np.array([0.8]) 
    T0 = 100
    N_test_QCELS = 10  #number of QCELS test
    T_list_QCELS = 61+T0/2*(np.arange(N_test_QCELS))
    print(T_list_QCELS)
    err_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    cost_list_avg_QCELS = np.zeros((len(p0_array),len(T_list_QCELS)))
    rate_success_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    max_T_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
    Navg = 50 #number of trying
    err_thres_hold=0.01
    err_thres_hold_QPE=0.01
    #-----------------------------    
    for a1 in range(len(p0_array)):
        p0=p0_array[a1]
        n_success_QCELS= np.zeros(len(T_list_QCELS))
        Total_time=np.zeros(len(T_list_QCELS))
        Variance=np.zeros(len(T_list_QCELS))
        Errs = np.zeros((Navg,len(T_list_QCELS)))
        for n_test in range(Navg):
            spectrum, population = generate_spectrum_population(eig, 
                    population, [p0])
            #------------------QCELS-----------------
            Nsample = 100

            for ix in range(len(T_list_QCELS)):
                T = T_list_QCELS[ix]
                NT = 5
                lambda_prior = spectrum[0]
                ground_energy_estimate_QCELS, cost_list_avg_QCELS[a1,ix], max_T_QCELS[a1,ix] = \
                        qcels_largeoverlap(spectrum, population, T, NT,
                                Nsample, lambda_prior)
                err_this_run_QCELS = np.abs(ground_energy_estimate_QCELS.x[2] - spectrum[0])
                # print(n_test,ix,err_this_run_QCELS)
                Errs[n_test][ix] = err_this_run_QCELS
                err_QCELS[a1,ix] = err_QCELS[a1,ix]+np.abs(err_this_run_QCELS)
                if np.abs(err_this_run_QCELS)<err_thres_hold:
                    n_success_QCELS[ix]+=1

                Total_time[ix] = Total_time[ix] + cost_list_avg_QCELS[a1,ix]
        
        for iy in range(len(T_list_QCELS)):
            Variance[iy] = func_variance(Errs[:,iy])

        rate_success_QCELS[a1,:] = n_success_QCELS[:]/Navg
        err_QCELS[a1,:] = err_QCELS[a1,:]/Navg
        Avg_total_cost = Total_time/Navg

    np.save('variance_err_qcels_g1.npy',Variance)
    np.save('average_err_qcels_g1.npy',err_QCELS[0,:])
    np.save('maximal_times_qcels_g1.npy',T_list_QCELS)
    np.save('total_times_qcels_g1.npy',Avg_total_cost)
