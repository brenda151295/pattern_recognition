import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

mean_array = np.array([[0, 2], [0, 0]])
covariance = [[
        [4., 1.8],
        [1.8, 1.],
    ],
    [
        [4., 1.8],
        [1.8, 1.],
    ]]
cmap_colors = ['Purples', 'Greens']
quantidade = [1500, 1500]

def generate_data(covariance):
    classes = []
    labels = []
    for i in range(2):
        mean = mean_array[i]
        arr = np.random.multivariate_normal(mean, covariance[i], quantidade[i]).T
        for j in range(len(arr[0])):
            point = [arr[0][j], arr[1][j]]
            classes.append(point)
            labels.append(i)
    return np.array(classes), np.array(labels)

np.random.seed(12)
data, labels = generate_data(covariance)
points_data = list(zip(*data))

data_test, labels_test = generate_data(covariance)
points_data_test = list(zip(*data_test))


#plt.suptitle('Train data', fontsize=16)
#plt.scatter(points_data[0], points_data[1], c = labels)
#plt.show()
#plt.suptitle('Test data', fontsize=16)
#plt.scatter(points_data_test[0], points_data_test[1], c=labels_test)
#plt.show()

clf = GaussianNB()
clf.fit(data, labels)
print ("Gaussian Naive Bayes accuracy:", clf.score(data_test, labels_test))

clf = LogisticRegression(random_state=0, solver='liblinear').fit(data, labels)
print ("Logistic Regression accuracy:", clf.score(data_test, labels_test))

covariance = [[
        [4., 1.8],
        [1.8, 1.],
    ],
    [
        [4., -1.8],
        [-1.8, 1.],
    ]]
np.random.seed(12)
data, labels = generate_data(covariance)
points_data = list(zip(*data))

data_test, labels_test = generate_data(covariance)
points_data_test = list(zip(*data_test))

plt.suptitle('Train data 2', fontsize=16)
plt.scatter(points_data[0], points_data[1], c = labels)
#plt.show()
plt.suptitle('Test data 2', fontsize=16)
plt.scatter(points_data_test[0], points_data_test[1], c=labels_test)
#plt.show()

clf = GaussianNB()
clf.fit(data, labels)
print ("(2) Gaussian Naive Bayes accuracy:", clf.score(data_test, labels_test))
print ("# Correct (class 0): ", list(clf.predict(data_test)[:1500]).count(0))
print ("# Correct (class 1): ", list(clf.predict(data_test)[1500:]).count(1))

clf = LogisticRegression(random_state=0, solver='liblinear').fit(data, labels)
print ("(2) Logistic Regression accuracy:", clf.score(data_test, labels_test))
print ("# Correct (class 0): ", list(clf.predict(data_test)[:1500]).count(0))
print ("# Correct (class 1): ", list(clf.predict(data_test)[1500:]).count(1))

C = [0.1, 0.18, 0.5, 0.8]
print ("(2) Logistic Regression Regularization")
for c in C:
	clf = LogisticRegression(random_state=0, solver='liblinear', C=c).fit(data, labels)
	print ("C = ",c)
	print ("Accuracy Test:", clf.score(data_test, labels_test))
	print ("Accuracy Train:", clf.score(data, labels))
	print ("# Correct (class 0): ", list(clf.predict(data_test)[:1500]).count(0))
	print ("# Correct (class 1): ", list(clf.predict(data_test)[1500:]).count(1))