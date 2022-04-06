import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import seaborn as sns
import numbers

from sklearn.neighbors import NearestNeighbors
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler)

from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as shuffle_


fig, ((plt1, plt2, plt3), (plt4,plt5, plt6)) = plt.subplots(2, 3, figsize=(15, 9))

def generate_points(n_samples=100, centers=2, random_state=None):

    generator = check_random_state(random_state)

    centers = check_array(centers)
    n_features = centers.shape[1]

    X = []
    y = []

    n_centers = centers.shape[0]
    
    n_samples_per_center = [int(n_samples // n_centers)] * n_centers
    for i in range(n_samples % n_centers):
        n_samples_per_center[i] += 1
    
    for i, (n, std) in enumerate(zip(n_samples_per_center, np.ones(len(centers)))):
        X.append(centers[i] + generator.normal(scale=std, size=(n, n_features)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    return X, y

n_samples = 10000
random_state = 170
centers = [(-9, -6), (-3, -3), (1, 0)]
X, y = generate_points(n_samples=n_samples, centers=centers, random_state=random_state)


for i in range(0, y.size):
    if y[i]==0:
        transformation = [[0.5, -0.25], [-0.5, 1]]
        X[i] = np.dot(X[i], transformation)
    if y[i]==1:
        transformation = [[0.5, 0], [-0.5, 1]]
        X[i] = np.dot(X[i], transformation)

n1=n_samples//40
n2=n_samples//50
n3=n_samples//160

plt1.set_title('Original data')
plt1.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')

X = np.vstack((X[y == 0][:n1], X[y == 1][:n2], X[y == 2][:n3]))


newy = np.concatenate((np.full((n1,1),0), np.full((n2,1),1), np.full((n3,1),2)))


colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in newy]		
plt2.set_title('Different density data')
plt2.scatter(X[:, 0], X[:, 1], marker='o',c=colors,  s=25, edgecolor='k')


sampler = RandomUnderSampler(random_state=0)
X_res, y_res = sampler.fit_resample(X, newy)
print(X.shape)
print(X_res.shape)
colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_res]
plt3.set_title('Undersampled data')
plt3.scatter(X_res[:, 0], X_res[:, 1], c=colors, linewidth=0.5, edgecolor='black')


NN = NearestNeighbors(n_neighbors=len(X)).fit(X)
distances, indices = NN.kneighbors(X)

print(distances)


plt4.set_title('minPts elbow')
plt4.plot(np.sort(distances[:, distances.shape[0]-1]))


##############################################
#----------------- PLT5 ----------------------
##############################################
#min_samples=np.log10(n_samples)



min_samples=7
epsCurrent = 0.01
epsStep = 0.01
epsMax = 0.05
pointsCurrent = X
pointClusters = np.zeros(pointsCurrent.size)
while epsCurrent < epsMax:
    print(epsCurrent)
    db = DBSCAN(eps=epsCurrent, min_samples=min_samples).fit(pointsCurrent)
    labels = db.labels_
    for i in range(0, pointsCurrent.size):
        if labels[i]!= -1:
            pointClusters[i]=pointsCurrent[i]
            np.delete(pointsCurrent, i)

    epsCurrent = epsCurrent + epsStep


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))
plt5.set_title('Different density data DBSCAN')
plt5.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')


##############################################
#----------------- PLT6 ----------------------
##############################################


eps=0.38
#min_samples=np.log10(n_samples)
min_samples=7
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_res)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))


plt6.set_title('US DBSCAN')
plt6.scatter(X_res[:, 0], X_res[:, 1], marker='o', c=labels, s=25, edgecolor='k')
plt.show()
#fig.savefig('dbscanGenUs2.jpg', dpi=100)