import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
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
'''ax.scatter3D(train_data[0][0], train_data[0][1], train_data[0][2], c=train_data[0][2], cmap=cmap_colors_train[0]);
ax.scatter3D(train_data[1][0], train_data[1][1], train_data[1][2], c=train_data[1][2], cmap=cmap_colors_train[1]);
ax.scatter3D(train_data[2][0], train_data[2][1], train_data[2][2], c=train_data[2][2], cmap=cmap_colors_train[2]);

ax.scatter3D(test_data[0][0], test_data[0][1], test_data[0][2], c=test_data[0][2], cmap=cmap_colors_test[0]);
ax.scatter3D(test_data[1][0], test_data[1][1], test_data[1][2], c=test_data[1][2], cmap=cmap_colors_test[1]);
ax.scatter3D(test_data[2][0], test_data[2][1], test_data[2][2], c=test_data[2][2], cmap=cmap_colors_test[2]);

plt.show()'''

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

media_1, covariance_1 = maximum_likelihood_gauss(test_data[0])
media_2, covariance_2 = maximum_likelihood_gauss(test_data[1])
media_3, covariance_3 = maximum_likelihood_gauss(test_data[2])


media_test = np.array([media_1, media_2, media_3])
#print ("RESULT EXERCISE 1: ", (covariance_1+covariance_2+covariance_3)/3)


def euclidean_distance(media_test, media_model):
    l, _, c = media_test.shape
    m, _, N = media_model.shape

    for i in range(N):
        l = []
        for j in range(c):
            dist = math.sqrt(math.pow((media_model[i][0][0]-media_test[j][0][0]),2) + math.pow((media_model[i][0][1]-media_test[j][0][1]),2) + math.pow((media_model[i][0][2]-media_test[j][0][2]),2))
            l.append(dist)
        print ("class", i+1, ":", min(l))
euclidean_distance(media_test, media_train)
