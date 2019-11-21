import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def generate_data(covariance, mean_array, quantidade, labels):
    classes = []
    labels_out = []
    for i in range(len(quantidade)):
        mean = mean_array[i]
        arr = np.random.multivariate_normal(mean, covariance[i], quantidade[i]).T
        for j in range(len(arr[0])):
            point = [arr[0][j], arr[1][j]]
            classes.append(point)
            if len(labels) > 1:
                labels_out.append(i)
            else:
                labels_out.append(labels[0])
    return np.array(classes), np.array(labels_out)

mean_array_1 = np.array([[0, 0], [10, 0], [0, 9], [9, 8]])
covariance_1 = [[
        [1.0, 0],
        [0, 1.0],
    ],
    [
        [1.0, 0.2],
        [0.2, 1.5],
    ],
    [
        [1.0, 0.4],
        [0.4, 1.1],
    ],
    [
        [0.3, 0.2],
        [0.2, 0.5],
    ]]
quantidade_1 = [100, 100, 100, 100]
X_train, labels_1 = generate_data(covariance_1, mean_array_1, quantidade_1, labels=[0,1,2,3])
points_data = list(zip(*X_train))

plt.suptitle('Train data', fontsize=16)
plt.scatter(points_data[0], points_data[1])
plt.show()
n_clusters = [4, 3, 5]
for n in n_clusters:
    kmeans = KMeans(n_clusters=n, random_state=None)
    y_pred = kmeans.fit_predict(X_train)
    centers = kmeans.cluster_centers_
    print (centers)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred)
    plt.scatter(centers[:, 0], centers[:, 1], c='r')
    plt.title("Clusters="+str(n))
    plt.show()

init = np.array([[-2,-2],[-2.1,-2.1],[-2,-2.2],[-2.1,-2.2]], np.float64)
kmeans = KMeans(n_clusters=4, random_state=None, init=init)
y_pred = kmeans.fit_predict(X_train)
centers = kmeans.cluster_centers_
print (centers)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred)
plt.scatter(centers[:, 0], centers[:, 1], c='r')
plt.title("Clusters = 4 - Centers initialized")
plt.show()
     