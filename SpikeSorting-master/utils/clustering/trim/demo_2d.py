import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import pandas as pd

from approximation_scheme import approximationScheme
from utils.dataset_parsing.clustering_datasets import getTINSData

fig, ((plt1, plt2, plt3), (plt4,plt5, plt6)) = plt.subplots(2, 3, figsize=(15, 9))

n_samples = 10000
random_state = 170
centers = [(-9, -6), (-3, -3), (1, 0)]
# X, y = fs.generate_overlapping_clusters(n_samples=n_samples, centers=centers, random_state=random_state)

# for i in range(0, y.size):
#     if y[i] == 0:
#         transformation = [[0.5, -0.25], [-0.5, 1]]
#         X[i] = np.dot(X[i], transformation)
#     if y[i] == 1:
#         transformation = [[0.5, 0], [-0.5, 1]]
#         X[i] = np.dot(X[i], transformation)


# Importing the dataset

X, _ = getTINSData()


plt1.set_title('Original data')
plt1.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')


plt1.set_title('Original data')
plt1.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')


newX = approximationScheme(X)

plt2.set_title('Rarefied data')
plt2.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')


NN1 = NearestNeighbors(n_neighbors=np.log(len(newX)).astype(int)).fit(newX)
distances1, indices1 = NN1.kneighbors(newX)

plt4.set_ylim(0, 1)
plt4.set_title('eps elbow')
plt4.plot(np.sort(distances1[:, distances1.shape[1]-1]), color='red')

eps=0.4
min_samples=np.log(len(newX))*10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(newX)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))
plt5.set_title('DBSCAN')
plt5.scatter(newX[:, 0], newX[:, 1], marker='o', c=labels, s=25, edgecolor='k')

plt.show()

plt.show()