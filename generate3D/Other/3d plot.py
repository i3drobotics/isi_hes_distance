# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import math as m
import scipy

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# For each set of style and range settings, plot n random points in the box

# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
xs = (336.5, 338, 339, 341, 342, 344, 345.25, 347, 350, 354, 358.75, 366.5, 377.75, 394.25, 420)
ys = (208.5, 206, 203, 200, 196, 191.75, 189, 186, 179, 169, 156.75, 139, 113.5, 74, 8)
zs = (1.50, 1.40, 1.30, 1.20, 1.10, 1.00, 0.95 ,0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20)

xd = (1.50, 1.40, 1.30, 1.20, 1.10, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40)
yd = (-0.073832, -0.067711, -0.063103, -0.054289, -0.047802, -0.039255, -0.033372, -0.027006, -0.020167, -0.013518, -0.006954, 0.000692)

def func4(x, c, d, e):
    return c*x**2+d*x**1+e

def func(a, b, x):
    return a*x + b

popt,pcov=curve_fit(func,xd,yd,maxfev=1000)
error = np.sqrt(np.diag(pcov))
print('a =',popt[0],'+/-',error[0])
print('b =',popt[1],'+/-',error[1])


