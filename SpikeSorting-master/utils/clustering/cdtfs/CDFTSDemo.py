import numpy as np
import pandas as pd


from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt

import CDFTS as ts

fig, ((plt1, plt2, plt3), (plt4,plt5, plt6), (plt7,plt8, plt9)) = plt.subplots(3, 3, figsize=(15, 9))

# Importing the dataset
data = pd.read_csv('../../../datasets/data.csv', skiprows=0)
f1 = data['F1'].values
f2 = data['F2'].values
f3 = data['F3'].values

print("Shape:")
print(data.shape)
print("\n")

c1 = np.array([f1]).T
c2 = np.array([f2]).T
c3 = np.array([f3]).T


plt1.set_xlim(-5, 5)
plt1.set_ylim(-15, 15)
plt1.set_title('C1 C2')
plt1.scatter(c1, c2, marker='.', s=25, edgecolor='k')

plt2.set_xlim(-5, 5)
plt2.set_ylim(-15, 15)
plt2.set_title('C1 C3')
plt2.scatter(c1, c3, marker='.', s=25, edgecolor='k')

plt3.set_xlim(-5, 5)
plt3.set_ylim(-15, 15)
plt3.set_title('C2 C3')
plt3.scatter(c2, c3, marker='.', s=25, edgecolor='k')





X = np.append(c1[1:10000], c3[1:10000], axis=1)

shiftedX = ts.CDFTS(X,0.2,0.005,10)


plt4.set_xlim(-5, 5)
plt4.set_ylim(-15, 15)
plt4.set_title('Original data')
plt4.scatter(X[:, 0], X[:, 1], marker='.', s=25, edgecolor='k')


plt5.set_xlim(0, 1)
plt5.set_ylim(0, 1)
plt5.set_title('Shifted data')
plt5.scatter(shiftedX[:, 0], shiftedX[:, 1], marker='.', s=25, edgecolor='k')



eps = 0.05
min_samples=int(np.log(len(shiftedX))*10)

NN = NearestNeighbors(n_neighbors=min_samples).fit(shiftedX)
distances, indices = NN.kneighbors(shiftedX)

plt7.set_ylim(0, 0.5)
plt7.set_title('eps elbow')
plt7.plot(np.sort(distances[:, distances.shape[1]-1]), color='red')


db = DBSCAN(eps=eps, min_samples=min_samples).fit(shiftedX)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt8.set_xlim(0, 1)
plt8.set_ylim(0, 1)
plt8.set_title('DBSCAN')
plt8.scatter(shiftedX[:, 0], shiftedX[:, 1], marker='o', c=labels, s=25, edgecolor='k')

plt.show()

