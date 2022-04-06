import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from utils.dataset_parsing.clustering_datasets import getGenData

X,y = getGenData(plotFig=True)

fig, ((plt1, plt2, plt3), (plt4,plt5, plt6), (plt7,plt8, plt9)) = plt.subplots(3, 3, figsize=(15, 9))


NN = NearestNeighbors(n_neighbors=np.log(X.size).astype(int)).fit(X)
distances, indices = NN.kneighbors(X)

plt2.set_title('eps elbow')
plt2.plot(np.sort(distances[:, distances.shape[1]-1]), color='red', label = 'Elbow')
plt2.legend()


# DBSCAN at 0.25
eps = 0.25
min_samples=np.log(len(X))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(eps)
print('Estimated number of clusters for eps: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt4.set_title('DBSCAN at '+str(eps))
plt4.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

# DBSCAN at 0.5
eps = 0.5
min_samples=np.log(len(X))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(eps)
print('Estimated number of clusters for eps: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt5.set_title('DBSCAN at '+str(eps))
plt5.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

# DBSCAN at 0.75
eps = 0.8
min_samples=np.log(len(X))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(eps)
print('Estimated number of clusters for eps: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt6.set_title('DBSCAN at '+str(eps))
plt6.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

# DBSCAN at 1
eps = 1
min_samples=np.log(len(X))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(eps)
print('Estimated number of clusters for eps: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt7.set_title('DBSCAN at '+str(eps))
plt7.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

from sklearn.preprocessing import scale
Xs = scale(X)
plt8.set_title('Scaled')
plt8.scatter(Xs[:, 0], Xs[:, 1], marker='o', s=25, edgecolor='k')

eps = 0.35
min_samples=np.log(len(Xs))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(Xs)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(eps)
print('Estimated number of clusters for eps: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt9.set_title('DBSCAN at '+str(eps))
plt9.scatter(Xs[:, 0], Xs[:, 1], marker='o', c=labels, s=25, edgecolor='k')

plt.tight_layout()
plt.show()

