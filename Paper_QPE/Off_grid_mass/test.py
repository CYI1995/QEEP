import cvxpy as cp
import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
import random
import source as srs 
import statistics

N = 100
X_axis = np.zeros(N)
Y1_axis = np.zeros(N)
Y2_axis = np.zeros(N)


for n in range(N):

    x = n + 10
    X_axis[n] = n+1 
    Y1_axis[n] = 1 + math.log(0.5*(x-1))
    Y2_axis[n] = math.pi*(math.log(x)/math.log(10))

plt.plot(X_axis,Y1_axis,label = '1 + ln(N-1/2)')
plt.plot(X_axis,Y2_axis,label = 'log^2 N')
plt.legend()
plt.show()
