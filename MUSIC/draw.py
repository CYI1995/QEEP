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

Original_Data = np.load('sdp_data.npy')
Y2 = np.load('inverse_rw.npy')
Y1 = np.load('rw.npy')
X2 = np.load('inverse_rw_xaxis.npy')
E = np.load('espirit.npy')

N = 201
L = 100
f1 = 0.128
f2 = 0.064
f3 = 0.032
f4 = 0.016
c1 = 1/3
c2 = 1/3
c3 = 1/3
c4 = 1/4
a1 = np.zeros(N,dtype = complex)
a2 = np.zeros(N,dtype = complex)
a3 = np.zeros(N,dtype = complex)
for i in range(N):
    a1[i] = (math.cos(2*math.pi*f1*(i - L)) + 1j*math.sin(2*math.pi*f1*(i - L)))
    a2[i] = (math.cos(2*math.pi*f2*(i - L)) + 1j*math.sin(2*math.pi*f2*(i - L)))
    a3[i] = (math.cos(2*math.pi*f3*(i - L)) + 1j*math.sin(2*math.pi*f3*(i - L)))


Y3 = c1*a1 + c2*a2 + c3*a3

X1_axis = np.load('sampled_times.npy')
Hadamard = np.load('Hadamard_data.npy')
L = X1_axis.size
Y1_axis = np.zeros(L)
for l in range(L):
    idx = int(X1_axis[l])
    Y1_axis[l] = Hadamard[l].real
    X1_axis[l] = X1_axis[l] - int(N/2) 

X2_axis = np.zeros(N)
Y2_axis = np.zeros(N)

for t in range(N):
    X2_axis[t] = t - int(N/2)
    Y2_axis[t] = Original_Data[t][N].real

M = 1000
X3_axis = np.zeros(M)
Y3_axis = np.zeros(M)
for m in range(M):
    t = N*(m+1)/M - int(N/2)
    val = 0
    val = val + c1*(math.cos(2*math.pi*f1*t))
    val = val + c2*(math.cos(2*math.pi*f2*t))
    val = val + c3*(math.cos(2*math.pi*f3*t))
    X3_axis[m] = t 
    Y3_axis[m] = val

F1x = np.zeros(M)
F2x = np.zeros(M)
F3x = np.zeros(M)
F1y = np.zeros(M)
F2y = np.zeros(M)


for m in range(M):
    F1x[m] = 0.032 
    F1y[m] = -0.1 + 1.2*(m+1)/M
    F2y[m] = 80*(m+1)/M
    F2x[m] = 0.064 
    F3x[m] = 0.128              

# plt.plot(X3_axis,Y3_axis,c = "k",label = 'true signal')
# plt.scatter(X2_axis,Y2_axis,c = 'b',marker = 'x',label = 'recovered signal')
# plt.scatter(X1_axis,Y1_axis,c = 'r',label = 'sampled points')
# plt.legend()
# plt.show()

fig, axs = plt.subplot_mosaic([['a)','a)'],['b)','c)']],
                              constrained_layout=True)

# fig, axs = plt.subplot_mosaic([['c)'], ['d)']],
#                               constrained_layout=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = matplotlib.transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')
    if(label == 'a)'):
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$y_t.real$')
        ax.plot(X3_axis,Y3_axis,c = "k",label = 'true signal')
        ax.scatter(X2_axis,Y2_axis,c = 'b',marker = 'x',label = 'recovered signal')
        ax.scatter(X1_axis,Y1_axis,c = 'r',label = 'sampled points')
        ax.legend(loc = 1)
    if(label == 'b)'):
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$C(k)$')
        ax.plot(X2,Y1, c = "k",marker = '.')
        ax.plot(F1x,F1y, c="r",label = 'true frequency')
        ax.plot(F2x,F1y,c="r")
        ax.plot(F3x,F1y,c="r")
        ax.set_ylim([0,1])
        ax.legend(loc = 4)
    if(label == 'c)'):
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$C^{-1}(k)$')
        ax.plot(X2,Y2, c = "k",marker = '.')
        ax.plot(F1x,F2y, c="r",label = 'true frequency')
        ax.plot(F2x,F2y,c="r")
        ax.plot(F3x,F2y,c="r")
        ax.set_ylim([0,80])
        ax.legend(loc = 1)

plt.show()