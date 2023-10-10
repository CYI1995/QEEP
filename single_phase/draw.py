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


X1 = np.load('grid_shift.npy')
Y1 = np.load('error.npy')
Y2 = np.load('solution.npy')
Noise = np.load('noise_signal.npy')

S = np.load('sampled_times.npy')
H = np.load('Hadamard_data.npy')

L = X1.size 
Y3 = np.zeros(L)
Y4 = np.zeros(L)
for l in range(L):
    Y2[l] = Y2[l]*1000
    Y3[l] = 20.25
    Y4[l] = 50*(0.344**2)


N = 1000
M = 50 
k = 20.25
X2 = np.arange(N)
Z1 = np.zeros(M)
Z2 = np.zeros(N)
for j in range(M):
    Z1[j] = H[j].real
for i in range(N):
    Z2[i] = math.cos(2*math.pi*k*i/N)

# plt.xlabel(r'$t$')
# plt.ylabel(r'$y_t.real$')
# plt.plot(X2,Z2, c = "k",label = 'recovered signal')
# plt.scatter(S,Z1,c = "r",label = 'sampled points')
# plt.legend()
# plt.show()

fig, axs = plt.subplot_mosaic([['a)','b)'],['c)','c)']],
                              constrained_layout=True)

# fig, axs = plt.subplot_mosaic([['c)'], ['d)']],
#                               constrained_layout=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = matplotlib.transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')
    if(label == 'a)'):
        ax.set_xlabel(r'$\nu$')
        ax.set_ylabel('total empirical error')
        ax.plot(X1,Y1, c = "k",marker = '.')
        ax.plot(X1,Y4, c = "b",label = r'$|\Omega|\eta^2$')
        ax.legend(loc = 1)
    if(label == 'b)'):
        ax.set_xlabel(r'$\nu$')
        ax.set_ylabel(r'$k_{\nu}$')
        ax.plot(X1,Y2, c = "k",marker = '.')
        ax.plot(X1,Y3, c = "r",label = r'$k_0$')
        ax.set_ylim(18,22)
        ax.legend()
    if(label == 'c)'):
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$y_t.real$')
        ax.plot(X2,Z2, c = "k",label = 'recovered signal')
        ax.scatter(S,Z1,c = "r",label = 'sampled points')
        ax.set_ylim(-1.1,1.1)
        ax.legend(loc = 1)

plt.show()