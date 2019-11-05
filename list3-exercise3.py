from scipy.optimize import minimize
import numpy as np

from numpy import array, dot
from qpsolvers import solve_qp


def main():

    G = np.array([[0.0001, 0], [0, 2]], dtype=np.double)
    a = np.array([1., 0.])
    C = np.array([[-1, -2], [-1, 0]], dtype=np.double)
    b = np.array([-2, -1], dtype=np.double)

    print ("QP solution:", solve_qp(G, a, C, b))

main()