from sklearn.datasets import make_circles
from sklearn.cluster import SpectralClustering
from matplotlib import pyplot
from pandas import DataFrame
from matplotlib import pyplot as plt

n_samples = 300
n_components = 2
X, y = make_circles(n_samples=400, factor=.5, noise=.05)


reds = y == 0
blues = y == 1
plt.figure()

plt.scatter(X[reds, 0], X[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()


clustering = SpectralClustering(n_clusters=2,  assign_labels='kmeans', gamma=2, coef0=1.5).fit_predict(X)

reds = clustering == 0
blues = clustering == 1
plt.figure()

plt.scatter(X[reds, 0], X[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()