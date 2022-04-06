import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_classification

from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss,
                                     InstanceHardnessThreshold,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     AllKNN,
                                     NeighbourhoodCleaningRule,
                                     OneSidedSelection)


def create_dataset(n_samples=1000, weights=(0.01, 0.01, 0.98), n_classes=3, class_sep=0.8, n_clusters=1, shift=0.0, hypercube=True):
    return make_classification(n_samples=n_samples, n_features=2,
								n_informative=2, n_redundant=0, n_repeated=0,
								n_classes=n_classes,
								n_clusters_per_class=n_clusters,
								weights=list(weights),
								class_sep=class_sep, 
								shift=shift,
								hypercube=hypercube,
								random_state=0)



fig, ((plt1, plt2, plt7), (plt3, plt4, plt8), (plt5, plt6, plt9)) = plt.subplots(3, 3, figsize=(10, 9))
X, y = create_dataset(n_samples=5000, weights=(0.01, 0.05, 0.94), class_sep=1)
transformation = [[0.5, -0.5], [-0.1, 1]]

plt7.set_title('Untransformed data')
plt7.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolor='k')
X = np.dot(X, transformation)
plt1.set_title('Initial data')
plt1.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolor='k')

				  

sampler = RandomUnderSampler(random_state=0)
X_res, y_res = sampler.fit_resample(X, y)
print(X.shape)
print(X_res.shape)
plt2.set_title('Undersampled data {}'.format(sampler.__class__.__name__))
plt2.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
fig.tight_layout()

#####################################################################

kmeans = KMeans(n_clusters=3)

# Fitting the input data
kmeans = kmeans.fit(X)

# Getting the cluster labels
labels = kmeans.predict(X)

# Centroid values
C = kmeans.cluster_centers_

plt3.set_title('KMeans on Original Data')
plt3.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')


#####################################################################
kmeans = KMeans(n_clusters=3)

# Fitting the input data
kmeans = kmeans.fit(X_res)

# Getting the cluster labels
labels = kmeans.predict(X_res)

# Centroid values
C = kmeans.cluster_centers_

plt4.set_title('KMeans on Undersampled Data')
plt4.scatter(X_res[:, 0], X_res[:, 1], marker='o', c=labels, s=25, edgecolor='k')


######################################################################
eps=0.3
min_samples=4
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

plt5.set_title('DBScan on Original Data')
plt5.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

######################################################################
eps=0.3
min_samples=4
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_res)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt6.set_title('DBScan on Undersampled Data')
plt6.scatter(X_res[:, 0], X_res[:, 1], marker='o', c=labels, s=25, label=labels, edgecolor='k')
plt.show()