from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

X = [[0,1,4,20,22,23],[1,0,3,22,24,25],[4,3,0,23,25,26],[20,22,23,0,3.5,3.5],[22,24,25,3.5,0,3.5],[23,25,26,3.6,3.7,0]]

Z = linkage(X, 'single')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()

Z = linkage(X, 'complete')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()
