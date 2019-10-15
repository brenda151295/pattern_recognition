import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
from scipy.spatial import distance
np.random.seed(12)

mean_array = np.array([[0.5, 0.5], [2.0, 2.0], [1.3, 1.8]])
covariance = [
        [1.2, 0.4],
        [0.4, 1.8],
    ]
cmap_colors = ['Purples', 'Blues', 'Greens']
quantidade = [300, 300, 300]

def generate_data():
    classes = []
    for i in range(3):
        mean = mean_array[i]
        classes.append(np.random.multivariate_normal(mean, covariance, quantidade[i]).T)    
    return classes

data = generate_data()
plt.scatter(data[0][0], data[0][1], cmap=cmap_colors[0], label="Class 1")
plt.scatter(data[1][0], data[1][1], cmap=cmap_colors[1], label="Class 2")
plt.scatter(data[2][0], data[2][1], cmap=cmap_colors[2], label="Class 3")

plt.legend(loc="lower right", frameon=False)
plt.show()