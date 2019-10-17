import numpy as np
from matplotlib import cm
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm
from sklearn import mixture
from random import randrange

from  sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# make the plot reproducible by setting the seed
np.random.seed(12)

mean_array = np.array([[1, 1], [3, 3], [2, 6]])
covariance_array = [[
        [0.1, 0],
        [0, 0.1],
    ],[
        [0.2, 0],
        [0, 0.2],
    ],[
        [0.3, 0],
        [0, 0.3],
    ],
    ]
P = [0.4, 0.4, 0.2]

n_samples = 1000

qtd = np.array([0, 0, 0])
while np.sum(qtd) < n_samples:
    R = randrange(100)
    if R > (P[0] + P[1])*100:
        qtd[2] = qtd[2] + 1
    elif R > P[0]*100:
        qtd[1] = qtd[1] + 1
    else:
        qtd[0] = qtd[0] + 1

print (qtd)

def generate_data(N):
    classes = []
    for i in range(N):
        mean = mean_array[i]
        covariance = covariance_array[i]
        classes.append(np.random.multivariate_normal(mean, covariance, qtd[i]).T)    
    return classes

classes = generate_data(N=3)
points_c1c2c3_separated = []
points_c1c2c3_tog = []
for classe in classes:
    temp = []
    for i in range(len(classe[0])):
        temp.append([classe[0][i], classe[1][i]])
        points_c1c2c3_tog.append([classe[0][i], classe[1][i]])
    points_c1c2c3_separated.append(temp)
points_c1c2c3_tog = np.asarray(points_c1c2c3_tog)

'''cmap_colors = ['Purples', 'Blues', 'Greens']
plt.scatter(classes[0][0], classes[0][1], cmap=cmap_colors[0], label="Class 1")
plt.scatter(classes[1][0], classes[1][1], cmap=cmap_colors[1], label="Class 2")
plt.scatter(classes[2][0], classes[2][1], cmap=cmap_colors[2], label="Class 3")

plt.legend(loc="lower right", frameon=False)
plt.show()'''

covariance_array_ini = [[
        [0.15, 0],
        [0, 0.15],
    ],[
        [0.27, 0],
        [0, 0.27],
    ],[
        [0.4, 0],
        [0, 0.4],
    ],
    ]
mean_array_ini = np.array([[0, 2], [5, 2], [5, 5]])
P_ini = [1/3., 1/3., 1/3.]
X = points_c1c2c3_tog

x,y = np.meshgrid(np.sort(points_c1c2c3_tog[:,0]),np.sort(points_c1c2c3_tog[:,1]))
XY = np.array([x.flatten(),y.flatten()]).T
covars = [0.15, 0.27, 0.4]

#GMM = GaussianMixture(n_components=3, means_init=mean_array_ini, reg_covar=[0.15, 0.27, 0.4]).fit(X) # Instantiate and fit the model
models = []
for covariance in covars:
    model = GaussianMixture(n_components=3, means_init=mean_array_ini,
              reg_covar=covariance, max_iter=20, random_state=0).fit(X)
    models.append(model)

for GMM in models:
    print('Converged:',GMM.converged_) # Check if the model has converged
    means = GMM.means_ 
    covariances = GMM.covariances_
    # Predict
    #Y = np.array([[0.5],[0.5]])
    #prediction = GMM.predict_proba(Y.T)
    #print(prediction)
    # Plot   
    fig = plt.figure(figsize=(10,10))
    ax0 = fig.add_subplot(111)
    ax0.scatter(points_c1c2c3_tog[:,0],points_c1c2c3_tog[:,1])
    #ax0.scatter(Y[0,:],Y[1,:],c='orange',zorder=10,s=100)
    for m,c in zip(means,covariances):
        print (m)
        print (c)
        multi_normal = multivariate_normal(mean=m,cov=c)
        ax0.contour(np.sort(points_c1c2c3_tog[:,0]),np.sort(points_c1c2c3_tog[:,1]),multi_normal.pdf(XY).reshape(len(X),len(X)),colors='black',alpha=0.3)
        ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
    
plt.show()
