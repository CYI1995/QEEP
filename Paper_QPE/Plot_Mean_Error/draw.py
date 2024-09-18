import cvxpy as cp
import numpy as np
import math
import random

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

times_cs = np.load('maximal_time_cs.npy')
times_mlqcels = np.load('maximal_times_mlqcels.npy')
times_mmqcels = np.load('maximal_times_mmqcels.npy')
times_qmegs = np.load('maximal_time_qmegs.npy')

cs_a8_ising = np.load('cs_a8_Ising.npy')
cs_a4_ising = np.load('cs_a4_Ising.npy')
cs_a2_ising = np.load('cs_a2_Ising.npy')

cs_a8_hubbard = np.load('cs_a8_hubbard.npy')
cs_a4_hubbard = np.load('cs_a4_hubbard.npy')
cs_a2_hubbard = np.load('cs_a2_hubbard.npy')

mlqcels_a8_ising = np.load('mlqcels_a8_Ising.npy')
mlqcels_a4_ising = np.load('mlqcels_a4_Ising.npy')
mlqcels_a2_ising = np.load('mlqcels_a2_Ising.npy')

mlqcels_a8_hubbard = np.load('mlqcels_a8_hubbard.npy')
mlqcels_a4_hubbard = np.load('mlqcels_a4_hubbard.npy')
mlqcels_a2_hubbard = np.load('mlqcels_a2_hubbard.npy')

mmqcels_a8_ising = np.load('mmqcels_a8_Ising.npy')
mmqcels_a4_ising = np.load('mmqcels_a4_Ising.npy')
mmqcels_a2_ising = np.load('mmqcels_a2_Ising.npy')

mmqcels_a8_hubbard = np.load('mmqcels_a8_hubbard.npy')
mmqcels_a4_hubbard = np.load('mmqcels_a4_hubbard.npy')
mmqcels_a2_hubbard = np.load('mmqcels_a2_hubbard.npy')

qmegs_a8_ising = np.load('qmegs_a8_Ising.npy')
qmegs_a4_ising = np.load('qmegs_a4_Ising.npy')
qmegs_a2_ising = np.load('qmegs_a2_Ising.npy')

qmegs_a8_hubbard = np.load('qmegs_a8_hubbard.npy')
qmegs_a4_hubbard = np.load('qmegs_a4_hubbard.npy')
qmegs_a2_hubbard = np.load('qmegs_a2_hubbard.npy')

x_left = 100
x_right = 1200
y_left = 0.00005
y_right = 0.005

fig, axs = plt.subplot_mosaic([['(a)','(b)'],['(c)','(d)'],['(e)','(f)']],
                              constrained_layout=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = matplotlib.transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')
    if(label == '(a)'):
        ax.set_title(r'$\alpha = 1/8$, Ising model')
        ax.set_xlabel(r'$T_{max}$')
        ax.set_ylabel('mean error')
        ax.plot(times_cs,cs_a8_ising,marker= 'o',c = "b",label = 'CS')
        ax.plot(times_mlqcels,mlqcels_a8_ising,c = "r",marker = 'x',label = 'ML-QCELS')
        ax.plot(times_mmqcels,mmqcels_a8_ising,c = "k",marker = 'd',label = 'MM-QCELS')
        ax.plot(times_qmegs,qmegs_a8_ising,marker= '^',c = "g",label = 'QMEGS')
        ax.set_xlim(x_left,x_right)
        ax.set_ylim(y_left,y_right)
        ax.loglog()
        ax.legend(loc = 1)
    if(label == '(b)'):
        ax.set_title(r'$\alpha = 1/8$, Hubbard model')
        ax.set_xlabel(r'$T_{max}$')
        ax.set_ylabel('mean error')
        ax.plot(times_cs,cs_a8_hubbard,marker= 'o',c = "b",label = 'CS')
        ax.plot(times_mlqcels,mlqcels_a8_hubbard,c = "r",marker = 'x',label = 'ML-QCELS')
        ax.plot(times_mmqcels,mmqcels_a8_hubbard,c = "k",marker = 'd',label = 'MM-QCELS')
        ax.plot(times_qmegs,qmegs_a8_hubbard,marker= '^',c = "g",label = 'QMEGS')
        ax.set_xlim(x_left,x_right)
        ax.set_ylim(y_left,y_right)
        ax.loglog()
        ax.legend(loc = 1)
    if(label == '(c)'):
        ax.set_title(r'$\alpha = 1/4$, Ising model')
        ax.set_xlabel(r'$T_{max}$')
        ax.set_ylabel('mean error')
        ax.plot(times_cs,cs_a4_ising,marker= 'o',c = "b",label = 'CS')
        ax.plot(times_mlqcels,mlqcels_a4_ising,c = "r",marker = 'x',label = 'ML-QCELS')
        ax.plot(times_mmqcels,mmqcels_a4_ising,c = "k",marker = 'd',label = 'MM-QCELS')
        ax.plot(times_qmegs,qmegs_a4_ising,marker= '^',c = "g",label = 'QMEGS')
        ax.set_xlim(x_left,x_right)
        ax.set_ylim(y_left,y_right)
        ax.loglog()
        ax.legend(loc = 1)
    if(label == '(d)'):
        ax.set_title(r'$\alpha = 1/4$, Hubbard model')
        ax.set_xlabel(r'$T_{max}$')
        ax.set_ylabel('mean error')
        ax.plot(times_cs,cs_a4_hubbard,marker= 'o',c = "b",label = 'CS')
        ax.plot(times_mlqcels,mlqcels_a4_hubbard,c = "r",marker = 'x',label = 'ML-QCELS')
        ax.plot(times_mmqcels,mmqcels_a4_hubbard,c = "k",marker = 'd',label = 'MM-QCELS')
        ax.plot(times_qmegs,qmegs_a4_hubbard,marker= '^',c = "g",label = 'QMEGS')
        ax.set_xlim(x_left,x_right)
        ax.set_ylim(y_left,y_right)
        ax.loglog()
        ax.legend(loc = 1)
    if(label == '(e)'):
        ax.set_title(r'$\alpha = 1/2$, Ising model')
        ax.set_xlabel(r'$T_{max}$')
        ax.set_ylabel('mean error')
        ax.plot(times_cs,cs_a2_ising,marker= 'o',c = "b",label = 'CS')
        ax.plot(times_mlqcels,mlqcels_a2_ising,c = "r",marker = 'x',label = 'ML-QCELS')
        ax.plot(times_mmqcels,mmqcels_a2_ising,c = "k",marker = 'd',label = 'MM-QCELS')
        ax.plot(times_qmegs,qmegs_a2_ising,marker= '^',c = "g",label = 'QMEGS')
        ax.set_xlim(x_left,x_right)
        ax.set_ylim(y_left,y_right)
        ax.loglog()
        ax.legend(loc = 1)
    if(label == '(f)'):
        ax.set_title(r'$\alpha = 1/2$, Hubbard model')
        ax.set_xlabel(r'$T_{max}$')
        ax.set_ylabel('mean error')
        ax.plot(times_cs,cs_a2_hubbard,marker= 'o',c = "b",label = 'CS')
        ax.plot(times_mlqcels,mlqcels_a2_hubbard,c = "r",marker = 'x',label = 'ML-QCELS')
        ax.plot(times_mmqcels,mmqcels_a2_hubbard,c = "k",marker = 'd',label = 'MM-QCELS')
        ax.plot(times_qmegs,qmegs_a2_hubbard,marker= '^',c = "g",label = 'QMEGS')
        ax.set_xlim(x_left,x_right)
        ax.set_ylim(y_left,y_right)
        ax.loglog()
        ax.legend(loc = 1)

plt.show()