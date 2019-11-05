import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import svm
from sklearn.svm import SVC

mean_array = np.array([[0, 0], [1.5, 1.5]])
covariance = [
        [0.2, 0],
        [0, 0.2],
    ]
cmap_colors = ['Purples', 'Greens']
quantidade = [200, 200]

def generate_data():
    classes = []
    labels = []
    for i in range(2):
        mean = mean_array[i]
        arr = np.random.multivariate_normal(mean, covariance, quantidade[i]).T
        for j in range(len(arr[0])):
            point = [arr[0][j], arr[1][j]]
            classes.append(point)
            labels.append(i)
    return np.array(classes), np.array(labels)

np.random.seed(10)
data, labels = generate_data()
points_data = list(zip(*data))

np.random.seed(100)
data_test, labels_test = generate_data()
points_data_test = list(zip(*data_test))

plt.scatter(points_data[0], points_data[1], cmap=cmap_colors[0], label="Train data")
plt.scatter(points_data_test[0], points_data_test[1], cmap=cmap_colors[0], label="Test data")


plt.legend(loc="lower right", frameon=False)
#plt.show()
C = [0.1, 0.2, 0.5, 1.0, 2.0, 20.0]
for c in C:
    clf = SVC(kernel='linear', tol=0.001, C=c)
    clf.fit(data, labels) 
    r = clf.score(data_test, labels_test, sample_weight=None)
    print ("C =", str(c), ":", r)
    print (len(clf.support_vectors_))
    
    #print (margin)
    w_norm = np.linalg.norm(clf.coef_)
    dist = 2. / w_norm
    print (dist)

    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin
    plt.figure(figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(data[:, 0], data[:, 1], c=labels, zorder=10, cmap=plt.cm.Paired,
                    edgecolors='k')

plt.show()