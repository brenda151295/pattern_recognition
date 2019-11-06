import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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
print (points)
print (labels)

points_data = list(zip(*points))
#plt.suptitle('Train data', fontsize=16)
plt.scatter(points_data[0], points_data[1], c = labels)
plt.show()