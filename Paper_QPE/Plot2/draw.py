import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import string

def sparse(array,L,k):

    M = int(L/k)
    new_array = np.zeros(M)
    for j in range(M):
        idx = (j+1)*k
        new_array[j] = array[idx-1]
    return new_array

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

Err1 = np.load('Average_err_r3.npy')
TT1 = np.load('Average_Total_time_r3.npy')
MT1 = np.load('Average_Maximal_time.npy')

Var_cs = (1/(math.sqrt(10)-1))*np.load('Variance_err_r3.npy')
Var_qcels = (1/(math.sqrt(10)-1))*np.load('variance_err_qcels.npy')

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
        ax.errorbar(MT1,Err1,Var_cs,marker= '.',c = "r",label = 'cs')
        ax.errorbar(MT_qcels[0:7],Err_qcels[0:7],Var_qcels[0:7], c = "k",marker = 'x',label = 'MM-qcels')
        ax.loglog()
        ax.legend(loc = 1) 
    if(label == 'b)'):
        ax.set_xlabel(r'$T_{\max}$')
        ax.set_ylabel(r'$T_{total}$')
        ax.plot(MT1,TT1, c = "r",marker = '.',label = 'cs')
        ax.plot(MT_qcels[0:7],TT_qcels[0:7], c = "k",marker = 'x',label = 'MM-qcels')
        ax.loglog()
        ax.legend(loc = 4)


plt.show()