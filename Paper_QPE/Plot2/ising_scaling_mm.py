import scipy.io as sio
import numpy as np
from scipy.linalg import toeplitz
from scipy.linalg import eigh
from copy import deepcopy
from scipy.optimize import minimize
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
from scipy.special import erf
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib
import fejer_kernel
import fourier_filter
import generate_cdf
import source as srs
from qcels import *
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['lines.markersize'] = 10

def func_variance(A):

    x2_mean = 0

    for i in range(len(A)):
        x2_mean = x2_mean + A[i]**2 

    return math.sqrt((x2_mean/len(A) - (np.sum(A)/len(A))**2))

eigenenergies = np.load('frequencies.npy')
population_raw = np.load('amplitudes.npy')

# plt.plot(eigenenergies,population_raw,'b-o');plt.show()

def spectrum_organize(Output_QCELS): #combine opposite r but same \theta
    dis_threshold=0.01
    L=len(Output_QCELS)
    L=int(L/3)
    index_list=np.zeros(L, dtype='float')
    weight_rearrange=np.zeros(L, dtype='complex')
    indicator=np.zeros(L, dtype='float')
    for l in range(L): #combine eigenvalues with oppposite weight
        if indicator[l]==0:
           weight_rearrange[l]=Output_QCELS[3*l]+1j*Output_QCELS[3*l+1]
           fix_energy_check=Output_QCELS[3*l+2]
           for j in range(l+1,L): 
               if np.abs(Output_QCELS[3*j+2]-fix_energy_check)<dis_threshold:
                  weight_rearrange[l]+=Output_QCELS[3*j]+1j*Output_QCELS[3*j+1]
                  indicator[j]=1
        indicator[l]=1
        weight_rearrange[l]=np.abs(weight_rearrange[l])
    index_list=sorted(range(L),key= lambda k:weight_rearrange[k], reverse=True)
    dominant_energy=np.zeros(L, dtype='float')
    for l in range(L):
        dominant_energy[l]=Output_QCELS[3*index_list[l]+2]
    return dominant_energy


p0_array = np.array([0.4], dtype='float') #initial overlap with the first eigenvector
p1_array = np.array([0.4], dtype='float') #initial overlap with the second eigenvector
N_test_QCELS = 10  #number of different circuit depths for QCELS test
#Generate T_max list
T_gap = 120
T_list_QCELS = T_gap*(2**(np.arange(N_test_QCELS)))### circuit depth for QCELS
print(T_list_QCELS)
#Array of error
err_QCELS_ground=np.zeros((len(p0_array),len(T_list_QCELS)))
err_QCELS_dominant=np.zeros((len(p0_array),len(T_list_QCELS)))
#Array of maximal running time
max_T_QCELS=np.zeros((len(p0_array),len(T_list_QCELS)))
#Array of total running time
cost_list_avg_QCELS = np.zeros((len(p0_array),len(T_list_QCELS)))
Navg = 10 #number of repetitions
#Failure threshold
err_threshold=0.01


Variance = []
Average_err = []
Total_time_set = []
Maximal_time_set = []


#-----------------------------    
for a1 in range(len(p0_array)):
    p0=p0_array[a1]
    p1=p1_array[a1]

    Errs = np.zeros((Navg,len(T_list_QCELS)))
    Variance = np.zeros(len(T_list_QCELS))
    for n_test in range(Navg):
        print("For p0,p1=",[p0,p1],"For N_test=",n_test+1)
        #Generate initial state with certain overlap
        spectrum, population = generate_spectrum_population(eigenenergies, 
                population_raw, [p0,p1])
        #random energy perturbation to avoid special case 
        # spectrum=spectrum+np.random.uniform(-1,1)*0.05
        #------------------QCELS-Gaussian-----------------
        gamma=1
        K=2
        for ix in range(len(T_list_QCELS)):
            T = T_list_QCELS[ix]
            T_0 = 10/(spectrum[1]-spectrum[0])
            NT_0 = 3000 #N_0
            NT = 2000 #N_{1:l}
            dominant_energy_estimate=np.zeros(K)
            Output_QCELS, cost_list_QCELS_this_run, max_T_QCELS_this_run = \
                    qcels_multimodal(spectrum, population, T_0, T, NT_0, NT, gamma, K, spectrum[0:2])#QCELS with time T
            dominant_energy_estimate=np.sort(spectrum_organize(Output_QCELS))
            err_QCELS_ground_this_run = np.abs(dominant_energy_estimate[0] - spectrum[0])
            err_QCELS_dominant_this_run = np.linalg.norm(dominant_energy_estimate - spectrum[0:2],np.inf)

            Errs[n_test][ix] = err_QCELS_dominant_this_run

            err_QCELS_ground[a1,ix] = err_QCELS_ground[a1,ix]+np.abs(err_QCELS_ground_this_run)
            err_QCELS_dominant[a1,ix] = err_QCELS_dominant[a1,ix]+np.abs(err_QCELS_dominant_this_run)
            max_T_QCELS[a1,ix]=max(max_T_QCELS[a1,ix],max_T_QCELS_this_run)
            cost_list_avg_QCELS[a1,ix]=cost_list_avg_QCELS[a1,ix]+cost_list_QCELS_this_run
            if np.abs(err_QCELS_dominant_this_run)>err_threshold: #failure case
                print('QCELS fail')

    for iy in range(len(T_list_QCELS)):
        Variance[iy] = func_variance(Errs[:,iy])

    err_QCELS_dominant[a1,:] = err_QCELS_dominant[a1,:]/Navg
    err_QCELS_ground[a1,:] = err_QCELS_ground[a1,:]/Navg
    cost_list_avg_QCELS[a1,:]=cost_list_avg_QCELS[a1,:]/Navg

np.save('variance_err_qcels.npy',Variance)
np.save('average_err_qcels.npy',err_QCELS_dominant[0,:])
np.save('maximal_times_qcels.npy',max_T_QCELS[0,:])
np.save('total_times_qcels.npy',cost_list_avg_QCELS[0,:])