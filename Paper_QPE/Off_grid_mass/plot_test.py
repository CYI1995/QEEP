import cvxpy as cp
import numpy as np
import math
import random
import source as srs
import matplotlib
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

Hadamard_err_ising = np.load('Hadamard_err_ising.npy')
Off_grid_mass_ising = np.load('Off_grid_mass_ising.npy')
Hadamard_err_hubbard = np.load('Hadamard_err_hubbard.npy')
Off_grid_mass_hubbard = np.load('Off_grid_mass_hubbard.npy')

N_T_total = 5
T_list = np.zeros(N_T_total)
for n_t in range(N_T_total):
    T_list[n_t] = int(100*1.4**(n_t+1))

# plt.title('Hubbard model')
# plt.xlabel(r'$T_{max}$')
# plt.plot(T_list,Hadamard_err_hubbard[0,:],linestyle = '--',marker = 'o',label = r'$\alpha = 1/8$, Hadamard error')
# plt.plot(T_list,Off_grid_mass_hubbard[0,:],marker = 'o',label = r'$\alpha = 1/8$, off-grid mass')
# plt.plot(T_list,Hadamard_err_hubbard[1,:],linestyle = '--',marker = 'x',label = r'$\alpha = 1/4$, Hadamard error')
# plt.plot(T_list,Off_grid_mass_hubbard[1,:],marker = 'x',label = r'$\alpha = 1/4$, off-grid mass')
# plt.plot(T_list,Hadamard_err_hubbard[2,:],marker = '^',linestyle = '--',label = r'$\alpha = 1/2$, Hadamard error')
# plt.plot(T_list,Off_grid_mass_hubbard[2,:],marker = '^',label = r'$\alpha = 1/2$, off-grid mass')
# plt.legend()
# plt.show()

plt.title('Ising model')
plt.xlabel(r'$T_{max}$')
plt.plot(T_list,Hadamard_err_ising[0,:],linestyle = '--',marker = 'o',label = r'$\alpha = 1/8$, Hadamard error')
plt.plot(T_list,Off_grid_mass_ising[0,:],marker = 'o',label = r'$\alpha = 1/8$, off-grid mass')
plt.plot(T_list,Hadamard_err_ising[1,:],linestyle = '--',marker = 'x',label = r'$\alpha = 1/4$, Hadamard error')
plt.plot(T_list,Off_grid_mass_ising[1,:],marker = 'x',label = r'$\alpha = 1/4$, off-grid mass')
plt.plot(T_list,Hadamard_err_ising[2,:],marker = '^',linestyle = '--',label = r'$\alpha = 1/2$, Hadamard error')
plt.plot(T_list,Off_grid_mass_ising[2,:],marker = '^',label = r'$\alpha = 1/2$, off-grid mass')
plt.legend()
plt.show()