import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
from scipy.spatial import distance
# make the plot reproducible by setting the seed
np.random.seed(12)

mean_array = np.array([[0, 0, 0], [1, 2, 2], [3, 3, 4]])
covariance = [
        [0.8, 0, 0],
        [0, 0.8, 0],
        [0, 0, 0.8],
    ]
# plot five blobs of data points
fig = plt.figure()
ax = fig.gca(projection='3d')
cmap_colors_train = ['Purples', 'Blues', 'Greens']
cmap_colors_test = ['Reds', 'Greys', 'Oranges']
quantidade = [333, 334, 333] #Equiprobable
def generate_data():
    classes = []
    for i in range(3):
        # generate normally distributed points to plot
        mean = mean_array[i]
        classes.append(np.random.multivariate_normal(mean, covariance, quantidade[i]).T)    
    return classes
        # plot them
        #surf = ax.scatter3D(X, Y, Z, c=Z, cmap=cmap_colors[i]);
    #plt.show()
train_data = generate_data()
test_data = generate_data()
'''ax.scatter3D(train_data[0][0], train_data[0][1], train_data[0][2], c=train_data[0][2], cmap=cmap_colors_train[0], label="Class 1");
ax.scatter3D(train_data[1][0], train_data[1][1], train_data[1][2], c=train_data[1][2], cmap=cmap_colors_train[1], label="Class 2");
ax.scatter3D(train_data[2][0], train_data[2][1], train_data[2][2], c=train_data[2][2], cmap=cmap_colors_train[2], label="Class 3");

ax.scatter3D(test_data[0][0], test_data[0][1], test_data[0][2], c=test_data[0][2], cmap=cmap_colors_test[0]);
ax.scatter3D(test_data[1][0], test_data[1][1], test_data[1][2], c=test_data[1][2], cmap=cmap_colors_test[1]);
ax.scatter3D(test_data[2][0], test_data[2][1], test_data[2][2], c=test_data[2][2], cmap=cmap_colors_test[2]);
plt.legend(loc="lower right", frameon=False)
plt.show()'''
points_c1c2c3 = []

class_1 = test_data[0]
class_2 = test_data[1]
class_3 = test_data[2]

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
    for i in range(N): #N=333
        sum_ += ((data[:,i] - media)* (data[:,i] - media).T)
    covariance = sum_/N
    return media, covariance

media_1, covariance_1 = maximum_likelihood_gauss(train_data[0])
media_2, covariance_2 = maximum_likelihood_gauss(train_data[1])
media_3, covariance_3 = maximum_likelihood_gauss(train_data[2])
media_train = np.array([media_1, media_2, media_3])
covariance_media = (covariance_1+covariance_2+covariance_3)/3

media_1, covariance_1 = maximum_likelihood_gauss(test_data[0])
media_2, covariance_2 = maximum_likelihood_gauss(test_data[1])
media_3, covariance_3 = maximum_likelihood_gauss(test_data[2])

media_test = np.array([media_1, media_2, media_3])

#print (media_train)
#print (media_test)

#print ("RESULT EXERCISE 1: ", (covariance_1+covariance_2+covariance_3)/3)




'''def euclidean_distance_X(data_test, media_model):
    m, _, N = media_model.shape
    for i in range(N):
            c = len(data_test[i][0])
            count = 0
            for j in range(c):
                distances = []
                for k in range(N):
                    dist = math.sqrt(math.pow(media_model[k][0][0]-data_test[i][0][j],2) +\
                        math.pow(media_model[k][0][1]-data_test[i][1][j],2) +\
                        math.pow(media_model[k][0][2]-data_test[i][2][j],2))
                    distances.append(dist)
                if i == distances.index(min(distances)):
                    count += 1
            print ("class", i+1, ":", count)

euclidean_distance_X(test_data, media_train)'''


'''def mahalanobis_distance_X(data_test, media_model, covariance):
    m, _, N = media_model.shape
    for i in range(N):
        c = len(data_test[i][0])
        count = 0
        for j in range(c):
            distances = []
            for k in range(N):
                point = [data_test[i][0][j], data_test[i][1][j], data_test[i][2][j]]
                dist = distance.mahalanobis(media_model[k][0], point, covariance)
                distances.append(dist)
            if i == distances.index(min(distances)):
                count += 1
        print ("class", i+1, ":", count)
mahalanobis_distance_X(test_data, media_train, covariance_media)
'''
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
        print ("class", cls+1, ":", count)

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
        print ("class", cls+1, ":", count)
mahalanobis_distance(points_c1c2c3, media_train, covariance_media)