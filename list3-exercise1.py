from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import scipy as sp
from sklearn.datasets.samples_generator import make_regression, make_blobs
from scipy.optimize import fmin


def himmelblau(params):
	X, Y = params
	print (X,Y)
	return (X*X+Y-11)**2 + (X+Y*Y-7)**2


fig = plt.figure()
ax = Axes3D(fig, azim = -29, elev = 49)
X = np.arange(-6, 6, 0.1)
Y = np.arange(-6, 6, 0.1)

X, Y = np.meshgrid(X, Y)

Z = (X*X+Y-11)**2 + (X+Y*Y-7)**2
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, norm = LogNorm(), cmap = cm.jet)

plt.xlabel("x")
plt.ylabel("y")

rcParams['font.size'] = 12

fig = plt.figure(figsize=(5, 5))
levels = np.logspace(0.3, 3.5, 15)
plt.contour(X, Y, Z, levels, cmap="viridis")
plt.xlabel(r"$x$", fontsize=14)
plt.ylabel(r"$y$", fontsize=14)
plt.xticks([-6, -3, 0, 3, 6])
plt.yticks([-6, -3, 0, 3, 6])
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.show()

#PONTOS TESTADOS
#x0 = [1.,1]
#x0 = [-1.,-3]
#x0 = [3.,-1]
x0 = [-1.,3]
xopt = fmin(himmelblau, x0=x0, xtol=1e-8)

