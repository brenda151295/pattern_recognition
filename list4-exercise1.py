import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import svm

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
data_test, labels = generate_data()

plt.scatter(points_data[0], points_data[1], cmap=cmap_colors[0], label="Train data")


plt.legend(loc="lower right", frameon=False)
#plt.show()

#clf = svm.SVC(gamma='scale', tol=0.001, C=0.1)
# figure number
fignum = 1

# fit the model
for name, penalty in (('unreg', 1), ('reg', 0.05)):
    clf = svm.SVC(kernel='linear', C=penalty)

    clf.fit(data, labels)

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(data[:, 0], data[:, 1], c=labels, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1

plt.show()