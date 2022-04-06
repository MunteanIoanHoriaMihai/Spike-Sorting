import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from utils.dataset_parsing.clustering_datasets import getTINSData

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

fig, ((plt1, plt2, plt3, plt4)) = plt.subplots(1, 4, figsize=(18, 9))

# Importing the dataset
X, _ = getTINSData()
print("Shape:", X.shape)
#X = StandardScaler().fit_transform(data)





# for no undersampling
NN1 = NearestNeighbors(n_neighbors=np.log(X.size).astype(int)*10).fit(X)
distances1, indices1 = NN1.kneighbors(X)
plt2.set_ylim(-0, 1)
plt2.set_title('eps elbow')
plt2.plot(np.sort(distances1[:, distances1.shape[1]-1]), color='red', label = 'NoUndersample')

eps=0.25
min_samples=np.log(len(X))*10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0],X[:, 1],X[:, 2], c=labels.astype(np.float))


eps=0.4
min_samples=np.log(len(X))*10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0],X[:, 1],X[:, 2],  c=labels.astype(np.float))
#ax.scatter(f1,f2,f3)




fig, ((plt1, plt2, plt3), (plt4,plt5, plt6), (plt7,plt8, plt9)) = plt.subplots(3, 3, figsize=(15, 9))
data = X[0:5000, (0, 2)]

NN = NearestNeighbors(n_neighbors=len(data)).fit(data)
distances, indices = NN.kneighbors(data)

# NN1 = NearestNeighbors(n_neighbors=np.log(X.size).astype(int)*10).fit(X)
#NN = NearestNeighbors(n_neighbors=int(np.log(len(X)))).fit(X)
#distances, indices = NN.kneighbors(X)

#plt4.set_ylim(0, 0.5)
#plt4.set_title('eps elbow')
#plt4.plot(np.sort(distances[:, distances.shape[1]-1]), color='red')



plt1.set_xlim(-5, 5)
plt1.set_ylim(-15, 15)
plt1.set_title('C1 C2')
plt1.scatter(X[:, 0], X[:, 1], marker='.', s=25, edgecolor='k')

plt2.set_xlim(-5, 5)
plt2.set_ylim(-15, 15)
plt2.set_title('C1 C3')
plt2.scatter(X[:, 0], X[:, 2], marker='.', s=25, edgecolor='k')

plt3.set_xlim(-5, 5)
plt3.set_ylim(-15, 15)
plt3.set_title('C2 C3')
plt3.scatter(X[:, 1], X[:, 2],marker='.', s=25, edgecolor='k')

plt5.set_xlim(-5, 5)
plt5.set_ylim(-15, 15)
plt5.set_title('DBSCAN')
plt5.scatter(data[:, 0], data[:, 1], marker='o', s=25, edgecolor='k')


eps = 0.35
min_samples=np.log(len(data))*10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt6.set_xlim(-5, 5)
plt6.set_ylim(-15, 15)
plt6.set_title('DBSCAN')
plt6.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')


# newX = np.zeros((len(X),2))
# k=0
# for i in range(0, len(X)):
# 	if distances[i][int(np.log(len(X)/2))-1] < eps:
# 		newX[k]=X[i]
# 		k=k+1
# newX = newX[:k]
# print(len(newX))

newX = np.zeros((len(data),2))
density = np.zeros(len(data))
flag = np.zeros(len(data))
k=0
avgDensity=0
for i in range(0, len(data)):
    for j in range(1, len(data)):
        if distances[i, j] < eps:
            k = k + 1
        else:
            break
    density[i] = k
    avgDensity += density[i]
avgDensity /= len(data)

for i in range(0, len(data)):
    if density[i] > avgDensity and random.random() < 0.5:
        flag[i] = 1

k = 0
for i in range(0, len(data)):
    if flag[i] == 0:
        newX[k] = data[i]
        k = k + 1
newX = newX[:k]
print(len(newX))




plt8.set_xlim(-5, 5)
plt8.set_ylim(-15, 15)
plt8.set_title('Reduced')
plt8.scatter(newX[:, 0], newX[:, 1], marker='o',s=25, edgecolor='k')

eps = 0.35
min_samples=np.log(len(newX))*10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(newX)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt9.set_xlim(-5, 5)
plt9.set_ylim(-15, 15)
plt9.set_title('DBSCAN')
plt9.scatter(newX[:, 0], newX[:, 1], marker='o', c=labels, s=25, edgecolor='k')


plt.show()
#fig.savefig('dbscan-'+str(eps)+'-'+str(min_samples)+'.jpg', dpi=100)


