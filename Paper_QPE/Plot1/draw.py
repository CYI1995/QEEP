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

# Err1 = np.load('Average_err_r3.npy')
# TT1 = np.load('Average_Total_time_r3.npy')
# Var1 = (1/(math.sqrt(50)-1))*np.load('Variance_err_r3.npy')

Err2 = np.load('Average_err_r2.npy')
TT2 = np.load('Average_Total_time_r2.npy')
Var2 = (1/(math.sqrt(50)-1))*np.load('Variance_err_r2.npy')

MT = np.load('Average_Maximal_time.npy')

Var_qcels = (1/(math.sqrt(50)-1))*np.load('variance_err_qcels.npy')
Err_qcels = np.load('average_err_qcels.npy')
MT_qcels = np.load('maximal_times_qcels.npy')
TT_qcels = np.load('total_times_qcels.npy')

fig, axs = plt.subplot_mosaic([['a)','b)']],
                              constrained_layout=True)

# fig, axs = plt.subplot_mosaic([['c)'], ['d)']],
#                               constrained_layout=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = matplotlib.transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')
    if(label == 'a)'):
        ax.set_xlabel(r'$T_{\max}$')
        ax.set_ylabel(r'$\epsilon$')
        # ax.errorbar(MT,Err1,Var1, marker= '.',c = "b",label = 'cs_r3')
        ax.errorbar(MT,Err2,Var2, marker= '.',c = "r",label = 'cs')
        ax.errorbar(MT_qcels,Err_qcels, Var_qcels, c = "k",marker = 'x',label = 'qcels')
        # ax.plot(MT,Scaling_1,c = "r", label = r'$0.06/T$')
        ax.loglog()
        ax.legend(loc = 1)
    if(label == 'b)'):
        ax.set_xlabel(r'$T_{\max}$')
        ax.set_ylabel(r'$T_{total}$')
        # ax.errorbar(TT1,Err1,Var1, c = "b",marker = '.',label = 'cs_r3')
        ax.errorbar(MT,TT2, c = "r",marker = '.',label = 'cs')
        ax.errorbar(MT_qcels,TT_qcels, c = "k",marker = 'x',label = 'qcels')
        # ax.plot(TT,Scaling_2, c = "r", label = r'$90/T$')
        ax.loglog()
        ax.legend(loc = 4)


plt.show()