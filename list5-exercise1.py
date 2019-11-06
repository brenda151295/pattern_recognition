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

points_test = []
labels_test = []
np.random.seed(100)
for i in range(150):
    point = np.random.uniform(-5,5, size=(1, 2))
    x = point[0][0]
    y = point[0][1]
    func = 0.05 * (x*x*x + x*x + x + 1)
    points_test.append([x,y])
    if func > y:
        labels_test.append(1)
    else:
        labels_test.append(-1)


points_data = list(zip(*points))
#plt.suptitle('Train data', fontsize=16)
i = range(-5, 5)
plt.plot(i, [0.05 * (x*x*x + x*x + x + 1) for x in i])
plt.scatter(points_data[0], points_data[1], c = labels)
plt.show()

points_data_test = list(zip(*points_test))
#plt.suptitle('Train data', fontsize=16)
plt.scatter(points_data_test[0], points_data_test[1], c = labels_test)
#plt.show()