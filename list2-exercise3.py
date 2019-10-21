import math
import numpy as np
from sklearn.neighbors.kde import KernelDensity
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
from random import randrange


np.random.seed(12)

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def generate_data(N, n_samples):
    quantidade = np.array([0, 0])

    while np.sum(quantidade) < n_samples:
        R = randrange(3)
        if R > 0:
            quantidade[1] = quantidade[1] + 1
        else:
            quantidade[0] = quantidade[0] + 1

    classes = []
    for i in range(N):
        mean = mean_array[i]
        arr = np.random.multivariate_normal(mean, covariance, quantidade[i]).T
        for j in arr[0]:
            classes.append(j)
    return np.array(classes)

mean_array = np.array([[0], [2]])
covariance = [
        [0.2],
    ]

data_1 = generate_data(N=2, n_samples = 500)
data_2 = generate_data(N=2, n_samples = 5000)

'''x_grid_1 = np.linspace(-4.5, 3.5, 500)
np.random.seed(0)

fig, ax = plt.subplots()

ax.plot(x_grid_1, kde_sklearn(data_1, x_grid_1, bandwidth=0.05),
            label='bw={0}'.format(0.05), linewidth=3, alpha=0.5)
ax.hist(data_1, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
ax.set_xlim(-5.5, 5.5)
ax.legend(loc='upper left')


fig, ax = plt.subplots()
x_grid_2 = np.linspace(-4.5, 3.5, 5000)
ax.plot(x_grid_2, kde_sklearn(data_2, x_grid_2, bandwidth=0.2),
            label='bw={0}'.format(0.2), linewidth=3, alpha=0.5)
ax.hist(data_2, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
ax.set_xlim(-5.5, 5.5)
ax.legend(loc='upper left')

plt.show()'''


data_3 = generate_data(N=2, n_samples = 1000)
k = 21

MU = np.zeros(2)
COV = np.eye(2)

def getData(size):
    data = np.random.multivariate_normal(MU, COV, size)
    return data

def area(data):
    size = 40
    X = []
    for i in range(len(data)):
        x = np.linspace(-4, 4, size)
        X.append(x)
    return np.array(X)

def knn(data, k):
    X = area(data)
    size = [len(X[0]), len(X[1])]
    print (size)
    knnpdf = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            x = np.array([X[0][i],X[1][j]])
            ds = [np.linalg.norm(x-y) for y in data]
            ds.sort()
            v = math.pi*ds[k-1]*ds[k-1]
            if v == 0:
                knnpdf[i,j] = 1
            else:
                knnpdf[i,j] = k/(1000*v)
    return X, knnpdf
   
n_set = [1, 16, 256, 10000]
fig = plt.figure()
pos = 1


data = data_3
X,P = knn(data, k)
#ax = fig.gca(projection='3d')
#ax = fig.add_subplot(len(n_set)/2, 2, pos, projection='3d')
#pos = pos + 1

fig, ax = plt.subplots()
ax.plot(X[0], P,
            label='bw={0}'.format(0.2), linewidth=3, alpha=0.5)

#px, py = np.meshgrid(X[0], X[1])
#ax.plot_surface(px, py, P, cmap='rainbow')
#ax.set_zlabel('Z')
#ax.set_ylabel('Y')
#ax.set_xlabel('X')
#ax.set_title(s)
        
plt.savefig("knn_result.png")
plt.show()
