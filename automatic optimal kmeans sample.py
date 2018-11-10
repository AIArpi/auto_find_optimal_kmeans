#from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()

# k means determine k
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Draw a linear function between the end point
# (y2 - y1 / x2 - x1) * x + c
linear = []
ld = len(distortions)
steep = (distortions[ld-1] - distortions[0]) / (ld - 1)
c = distortions[ld-1] - steep * ld
for x in range(0,ld):
    linear.append(steep * (x+1) + c)
    
plt.style.use('seaborn')
plt.plot(K, distortions, color = 'blue')
plt.plot(K, linear, color = 'red')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Look for max distance from linear ... between end points
distances = np.array(linear)-np.array(distortions)
max_index = distances.argmax(axis=0)+1

print('Optimal cluster number: ',max_index)

