from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('../../datasets/data.csv', skiprows=0)

f1 = data['F1'].values
f2 = data['F2'].values
f3 = data['F3'].values

print("Shape:", data.shape)
print("\n")
print("Head:")
print(data.head())
print("\n")
print(data.isna().sum())

kmeans = KMeans(n_clusters=4)

# Fitting the input data
kmeans = kmeans.fit(data)

# Getting the cluster labels
labels = kmeans.predict(data)
labels = kmeans.labels_

# Centroid values
C = kmeans.cluster_centers_

print("Cluster Centers:", C)
print("\n")

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(f1, f2, f3, c=labels.astype(np.float))
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
plt.show()
# fig.savefig('cluster4.jpg', dpi=100)

# make 3-class dataset for classification
# centers = [[-5, 0], [0, 1.5], [5, -1]]
# X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)

# centers = [(-4, -6), (-1, 1), (-7, 5)]
# X, y = make_blobs(n_samples=50000, n_features=2, cluster_std=1.0, centers=centers, shuffle=False, random_state=30)

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.5, -0.5], [-0.5, 1]]
X = np.dot(X, transformation)

plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')
plt.show()

kmeans = KMeans(n_clusters=3)

# Fitting the input data
kmeans = kmeans.fit(X)

# Getting the cluster labels
labels = kmeans.predict(X)

# Centroid values
C = kmeans.cluster_centers_

print("Cluster Centers:", C)

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')
plt.show()
# fig.savefig('kmeans.jpg', dpi=100)