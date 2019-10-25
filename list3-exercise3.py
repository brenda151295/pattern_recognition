from scipy.optimize import minimize

fun = lambda x: (x[0]+x[1]*x[1])

import numpy as np
import scipy.optimize
import scipy.stats
from quadprog import solve_qp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib import rcParams

def solve_qp_scipy(G, a, C, b, meq=0):
    # Minimize     1/2 x^T G x - a^T x
    # Subject to   C.T x >= b
    def f(x):
        return 0.5 * np.dot(x, G).dot(x) - np.dot(a, x)

    if C is not None and b is not None:
        constraints = [{
            'type': 'ineq',
            'fun': lambda x, C=C, b=b, i=i: (np.dot(C.T, x) - b)[i]
        } for i in range(C.shape[1])]
    else:
        constraints = []

    result = scipy.optimize.minimize(
        f, x0=np.zeros(len(G)), method='COBYLA', constraints=constraints,
        tol=1e-10, options={'maxiter': 2000})
    return result


def verify(G, a, C=None, b=None):
    xf, f, xu, iters, lagr, iact = solve_qp(G, a, C, b)
    result = solve_qp_scipy(G, a, C, b)
    np.testing.assert_array_almost_equal(result.x, xf)
    np.testing.assert_array_almost_equal(result.fun, f)
    print (result)

def main():
    
    fig = plt.figure()
    ax = Axes3D(fig, azim = -29, elev = 49)
    X = np.arange(-6, 6, 0.1)
    Y = np.arange(-6, 6, 0.1)

    X, Y = np.meshgrid(X, Y)

    Z = X + Y**2
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.jet)

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

    G = np.array([[1, 0], [0, 2]], dtype=np.double)
    a = np.array([1., 0.])
    C = np.array([[1, 2], [1, 0]], dtype=np.double)
    b = np.array([2, 1], dtype=np.double)
    verify(G, a, C, b)
main()