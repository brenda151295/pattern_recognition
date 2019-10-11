import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# make the plot reproducible by setting the seed
np.random.seed(12)

mean_array = [[0, 0, 0], [1, 2, 2], [3, 3, 4]]
covariance = [
        [0.8, 0, 0],
        [0, 0.8, 0],
        [0, 0, 0.8],
    ]
# plot five blobs of data points
fig = plt.figure()
ax = fig.gca(projection='3d')
cmap_colors = ['Purples', 'Blues', 'Greens']
quantidade = [340, 320, 340] #Equiprobable
def generate_data():
    for i in range(3):
        # generate normally distributed points to plot
        mean = mean_array[i]
        a= np.random.multivariate_normal(mean, covariance, quantidade[i]).T
        print (a)
        print(len(a))
        print(len(a[0]))
        input()
        # plot them
        #surf = ax.scatter3D(X, Y, Z, c=Z, cmap=cmap_colors[i]);
    #plt.show()
generate_data()
#def maximum_likelihood_gauss():
