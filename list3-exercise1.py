from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import scipy as sp
from sklearn.datasets.samples_generator import make_regression, make_blobs
import minhelper

def gradient_descent1(x, y, iters, alpha):
    costs = []
    predictions = []
    n = y.size 
    theta = np.random.rand(2)
    history = [theta] 
    
    for i in range(iters):
        prediction = np.dot(x, theta)
        if i % 10 == 0: predictions.append(prediction)
        
        error = prediction - y 
        cost = np.sum(error ** 2) / (2 * n)
        costs.append(cost)

        gradient = x.T.dot(error)/n 
        theta = theta - alpha * gradient  
        history.append(theta)

    return history, costs, predictions

fig = plt.figure()
ax = Axes3D(fig, azim = -29, elev = 49)
X = np.arange(-6, 6, 0.1)
Y = np.arange(-6, 6, 0.1)

'''X, Y = np.meshgrid(X, Y)

Z = (X*X+Y-11)**2 + (X+Y*Y-7)**2
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, norm = LogNorm(), cmap = cm.jet)

plt.xlabel("x")
plt.ylabel("y")

#plt.savefig("Himmelblau function.svg")

#plt.show()'

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
#plt.savefig("Himmelblau_contour.svg", bbox_inches="tight")
#plt.show()
'''