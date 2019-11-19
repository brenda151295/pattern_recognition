import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.svm import SVC

def plot_svc_decision_function(model, ax=None, plot_support=True, title=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors=['k','r','k'],
               levels=[-.5, 0, .5], alpha=0.8,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    if title is not None:
        plt.suptitle(title, fontsize=16)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

N = 150
points = []
labels = []
np.random.seed(0)
for i in range(150):
    point = np.random.uniform(-5,5, size=(1, 2))
    x = point[0][0]
    y = point[0][1]
    func = 0.05 * (x*x*x + x*x + x + 1)
    points.append([x,y])
    if func > y:
        labels.append(1)
    else:
        labels.append(-1)
points = np.array(points)
labels = np.array(labels)

points_test = []
labels_test = []
np.random.seed(100)
for i in range(150):
    point = np.random.uniform(-5,5, size=(1, 2))
    x = point[0][0]
    y = point[0][1]
    func = 0.05 * (x*x*x + x*x + x + 1)
    points_test.append([x,y])
    if func > y:
        labels_test.append(1)
    else:
        labels_test.append(-1)
points_test = np.array(points_test)
labels_test = np.array(labels_test)

points_data = list(zip(*points))
'''plt.suptitle('Train data', fontsize=16)
i = range(-5, 5)
plt.plot(i, [0.05 * (x*x*x + x*x + x + 1) for x in i])
plt.scatter(points_data[0], points_data[1], c = labels)
#plt.show()

points_data_test = list(zip(*points_test))
plt.suptitle('Test data', fontsize=16)
plt.scatter(points_data_test[0], points_data_test[1], c = labels_test)
#plt.show()'''


print ("Exercise 1b")
clf = SVC(kernel='linear', tol=0.001, C=2)
clf.fit(points, labels) 

r = clf.score(points, labels, sample_weight=None)
print ("Train accuracy C = 2:", r)
r = clf.score(points_test, labels_test, sample_weight=None)
print ("Test accuracy C = 2:", r)

print ("# support vectors = ", len(clf.support_vectors_))

#print (margin)
w_norm = np.linalg.norm(clf.coef_)
dist = 2. / w_norm
#print (dist)

'''margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
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
plt.scatter(points[:, 0], points[:, 1], c=labels, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

plt.show()
'''
print ("Exercise 1c")
gamma=[0.1, 2.0]
for g in gamma:
    clf = SVC(kernel='rbf', tol=0.001, C=2, gamma=1./g)
    clf.fit(points, labels) 
    r = clf.score(points, labels, sample_weight=None)
    print ("Train accuracy - gamma =", str(g), ":", r)
    r = clf.score(points_test, labels_test, sample_weight=None)
    print ("Test accuracy - gamma =", str(g), ":", r)
    print ("# support vectors = ", len(clf.support_vectors_))

    '''plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    plt.scatter(points_test[:, 0], points_test[:, 1], c=labels_test, zorder=10)
    plot_svc_decision_function(clf, title='SVM - RBF / Sigma = '+str(g))
    plt.show()'''

print ("Exercise 1d")
print ("SVM - RBF")
C = [0.2, 20, 200]
for c in C:
    clf = SVC(kernel='rbf', tol=0.001, C=c, gamma=1/1.5)
    clf.fit(points, labels) 
    
    r = clf.score(points, labels, sample_weight=None)
    print ("Train accuracy - C =", str(c), ":", r)
    r = clf.score(points_test, labels_test, sample_weight=None)
    print ("Test accuracy - C =", str(c), ":", r)
    print ("# support vectors = ", len(clf.support_vectors_))

    '''plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    plt.scatter(points_test[:, 0], points_test[:, 1], c=labels_test, zorder=10)

    plot_svc_decision_function(clf, title='SVM - RBF / C = '+str(c))

    plt.show()'''

print ("SVM - Polynomial")

for c in C:
    clf = SVC(kernel='poly', tol=0.001, C=c, degree=3, coef0=1)
    clf.fit(points, labels) 
    r = clf.score(points, labels, sample_weight=None)
    print ("Train accuracy - C =", str(c), ":", r)
    r = clf.score(points_test, labels_test, sample_weight=None)
    print ("Test accuracy - C =", str(c), ":", r)
    print ("# support vectors = ", len(clf.support_vectors_))

    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    plt.scatter(points_test[:, 0], points_test[:, 1], c=labels_test, zorder=10)

    plot_svc_decision_function(clf, title='SVM - Polynomial / C = '+str(c))
    plt.show()
