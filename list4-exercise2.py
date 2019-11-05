import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.naive_bayes import GaussianNB

mean_array = np.array([[0, 2], [0, 0]])
covariance = [
        [4., 1.8],
        [1.8, 1.],
    ]
cmap_colors = ['Purples', 'Greens']
quantidade = [1500, 1500]

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


plt.suptitle('Train data', fontsize=16)
plt.scatter(points_data[0], points_data[1], c = labels)
#plt.show()
plt.suptitle('Test data', fontsize=16)
plt.scatter(points_data_test[0], points_data_test[1], c=labels_test)
#plt.show()

clf = GaussianNB()
clf.fit(data, labels)
print ("Gaussian Naive Bayes accuracy:", clf.score(data_test, labels_test))