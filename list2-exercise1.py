import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
from scipy.spatial import distance
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import make_blobs, make_moons, make_regression, load_iris
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from  sklearn.model_selection import train_test_split
# make the plot reproducible by setting the seed
np.random.seed(12)

mean_array = np.array([[0, 0, 0], [1, 2, 2], [3, 3, 4]])
covariance = [
        [0.8, 0, 0],
        [0, 0.8, 0],
        [0, 0, 0.8],
    ]
fig = plt.figure()
ax = fig.gca(projection='3d')
cmap_colors_train = ['Purples', 'Blues', 'Greens']
cmap_colors_test = ['Reds', 'Greys', 'Oranges']

n_samples = 1000
quantidade = np.array([0, 0, 0])

while np.sum(quantidade) < n_samples:
    R = randrange(3)
    if R == 2:
        quantidade[2] = quantidade[2] + 1
    elif R == 1:
        quantidade[1] = quantidade[1] + 1
    else:
        quantidade[0] = quantidade[0] + 1


def generate_data(N):
    classes = []
    for i in range(N):
        mean = mean_array[i]
        classes.append(np.random.multivariate_normal(mean, covariance, quantidade[i]).T)    
    return classes

train_data = generate_data(N=3)
test_data = generate_data(N=3)

ax.scatter3D(train_data[0][0], train_data[0][1], train_data[0][2], c=train_data[0][2], cmap=cmap_colors_train[0], label="Class 1");
ax.scatter3D(train_data[1][0], train_data[1][1], train_data[1][2], c=train_data[1][2], cmap=cmap_colors_train[1], label="Class 2");
ax.scatter3D(train_data[2][0], train_data[2][1], train_data[2][2], c=train_data[2][2], cmap=cmap_colors_train[2], label="Class 3");

ax.scatter3D(test_data[0][0], test_data[0][1], test_data[0][2], c=test_data[0][2], cmap=cmap_colors_test[0]);
ax.scatter3D(test_data[1][0], test_data[1][1], test_data[1][2], c=test_data[1][2], cmap=cmap_colors_test[1]);
ax.scatter3D(test_data[2][0], test_data[2][1], test_data[2][2], c=test_data[2][2], cmap=cmap_colors_test[2]);
plt.legend(loc="lower right", frameon=False)
plt.show()

points_c1c2c3 = []

class_1 = test_data[0]
class_2 = test_data[1]
class_3 = test_data[2]

#Colocando os pontos em formato de ponto
points_c1 = []
for j in range(len(class_1[0])):
    points_c1.append([class_1[0][j], class_1[1][j], class_1[2][j]])
points_c1c2c3.append(points_c1)

points_c2 = []
for j in range(len(class_2[0])):
    points_c2.append([class_2[0][j], class_2[1][j], class_2[2][j]])
points_c1c2c3.append(points_c2)

points_c3 = []
for j in range(len(class_3[0])):
    points_c3.append([class_3[0][j], class_3[1][j], class_3[2][j]])
points_c1c2c3.append(points_c3)


def maximum_likelihood_gauss(data):
    l, N = data.shape
    media = np.zeros((1,l))
    for i in range(N):
        media += data[:,i]
    media = media/N
    sum_ = np.zeros((l,l))
    for i in range(N): #N=333 ou 334 ou 333
        sum_ += ((data[:,i] - media)* (data[:,i] - media).T)
    covariance = sum_/N
    return media, covariance

media_1, covariance_1_train = maximum_likelihood_gauss(train_data[0])
media_2, covariance_2_train = maximum_likelihood_gauss(train_data[1])
media_3, covariance_3_train = maximum_likelihood_gauss(train_data[2])
media_train = np.array([media_1, media_2, media_3])
covariance_media = (covariance_1_train+covariance_2_train+covariance_3_train)/3

media_1, covariance_1 = maximum_likelihood_gauss(test_data[0])
media_2, covariance_2 = maximum_likelihood_gauss(test_data[1])
media_3, covariance_3 = maximum_likelihood_gauss(test_data[2])

media_test = np.array([media_1, media_2, media_3])
def euclidean_distance(data_test, media_model):
    m, _, N = media_model.shape
    for cls in range(N):
        points = data_test[cls]
        count = 0
        for point in points:
            distances = []
            for k in range(N):
                media = media_model[k]
                dist = math.sqrt(\
                        math.pow(media[0][0]-point[0],2) +\
                        math.pow(media[0][1]-point[1],2) +\
                        math.pow(media[0][2]-point[2],2))
                distances.append(dist)
            if cls == distances.index(min(distances)):
                count += 1
        print ("Classe", cls+1, ":", count, (333-count)/333)

print ("ACERTOS - EUCLIDEAN DISTANCE")
euclidean_distance(points_c1c2c3, media_train)

def mahalanobis_distance(data_test, media_model, covariance):
    m, _, N = media_model.shape
    for cls in range(N):
        points = data_test[cls]
        count = 0
        for point in points:
            distances = []
            for k in range(N):
                media = media_model[k]
                dist = distance.mahalanobis(media[0], point, covariance)
                distances.append(dist)
            if cls == distances.index(min(distances)):
                count += 1
        print ("Classe", cls+1, ":", count, (333-count)/333)

print ("ACERTOS - AHALANOBIS")
mahalanobis_distance(points_c1c2c3, media_train, covariance_media)
def pred(test_data, media, covariance):
    n_classes = 3
    prob_vect = np.zeros(3)
    for i in range(n_classes):
        mnormal = multivariate_normal(mean=media[i], cov=covariance[i])
        # We use uniform priors
        prior = 1./n_classes
        
        prob_vect[i] = prior*mnormal.pdf(test_data)
        sumatory = 0.
        for j in range(n_classes):
            mnormal = multivariate_normal(mean=media[j], cov=covariance[j])
            sumatory += prior*mnormal.pdf(test_data)
        prob_vect[i] = prob_vect[i]/sumatory
    return prob_vect

print ("ACERTOS - BAYES CLASSIFIER")
media_train = np.array([media_train[0][0], media_train[1][0], media_train[2][0]])
covariance = np.array([covariance_1_train, covariance_2_train, covariance_3_train])

hit = 0
for point in points_c1:
    ypred = pred(point, media_train, covariance)
    if np.argmax(ypred) == 0:
        hit += 1
print ("Classe 1:", hit, (333-hit)/333)

hit = 0
for point in points_c2:
    ypred = pred(point, media_train, covariance)
    if np.argmax(ypred) == 1:
        hit += 1
print ("Classe 2:", hit, (334-hit)/333)

hit = 0
for point in points_c3:
    ypred = pred(point, media_train, covariance)
    if np.argmax(ypred) == 2:
        hit += 1
print ("Classe 3:", hit, (333-hit)/333)