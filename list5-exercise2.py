import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn import tree

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

mean_array_1 = np.array([[0, 3], [11, -1]])
covariance_1 = [[
        [0.2, 0],
        [0, 2.0],
    ],
    [
        [3.0, 0],
        [0, 0.5],
    ]]
quantidade_1 = [500, 500]

mean_array_2 = np.array([[3, -2], [7.5, 4]])
covariance_2 = [[
        [5.0, 0],
        [0, 0.5],
    ],
    [
        [7.0, 0],
        [0, 0.5],
    ]]
quantidade_2 = [500, 500]

mean_array_3 = np.array([[7, 2]])
covariance_3 = [[
        [8.0, 0],
        [0, 0.5],
    ]
    ]
quantidade_3 = [500]


np.random.seed(0)
data_1, labels_1 = generate_data(covariance_1, mean_array_1, quantidade_1, labels=[0])
data_2, labels_2 = generate_data(covariance_2, mean_array_2, quantidade_2, labels=[1])
data_3, labels_3 = generate_data(covariance_3, mean_array_3, quantidade_3, labels=[2])

X_train = np.concatenate((data_1, data_2, data_3), axis=0)
Y_train = np.concatenate((labels_1, labels_2, labels_3), axis=0)

np.random.seed(100)
data_1, labels_1 = generate_data(covariance_1, mean_array_1, quantidade_1, labels=[0])
data_2, labels_2 = generate_data(covariance_2, mean_array_2, quantidade_2, labels=[1])
data_3, labels_3 = generate_data(covariance_3, mean_array_3, quantidade_3, labels=[2])

X_test = np.concatenate((data_1, data_2, data_3), axis=0)
Y_test = np.concatenate((labels_1, labels_2, labels_3), axis=0)

points_data = list(zip(*X_train))
points_data_test = list(zip(*X_test))

plt.suptitle('Train data', fontsize=16)
plt.scatter(points_data[0], points_data[1], c = Y_train)
plt.show()
plt.suptitle('Test data', fontsize=16)
plt.scatter(points_data_test[0], points_data_test[1], c = Y_test)
plt.show()


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
tree.plot_tree(clf) 
#plt.show()

r = clf.score(X_train, Y_train)
print ("Train accuracy:", r)
r = clf.score(X_test, Y_test)
print ("Test accuracy:", r)