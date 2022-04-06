import matplotlib.gridspec as gridspec
from matplotlib import colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from SciOptics import OPTICS
from utils.dataset_parsing.clustering_datasets import getGenData

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors.pop('w')
colors.pop('k')
print(colors)

X, y = getGenData()

clust = OPTICS(min_samples=9, rejection_ratio=0.7)

# Run the fit
clust.fit(X)

_, labels_025 = clust.extract_dbscan(0.25)
_, labels_05 = clust.extract_dbscan(0.5)
_, labels_075 = clust.extract_dbscan(0.75)
print(len(set(clust.labels_)),len(set(labels_025)),len(set(labels_075)))


space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :2])
ax2 = plt.subplot(G[0, 2])
ax3 = plt.subplot(G[1, 0])
ax4 = plt.subplot(G[1, 1])
ax5 = plt.subplot(G[1, 2])

#colors = ['g.', 'r.', 'b.', 'y.', 'c.', 'aqua.', 'greenyellow.','indigo.','honeydew.','orangered.','yellow.','darkred.','darkslateblue.','darkturquoise.']
# Reachability plot
for k, c in zip(range(0, len(set(clust.labels_))), colors):
    Xk = space[labels == k]
    Rk = reachability[labels == k]
    ax1.plot(Xk, Rk, c, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

# OPTICS
for k, c in zip(range(0, len(set(clust.labels_))), colors):
    Xk = X[clust.labels_ == k]
    ax2.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3)
ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k.', alpha=0.1)
ax2.set_title('Automatic Clustering\nOPTICS')

# DBSCAN at 0.25
for k, c in zip(range(0, len(set(labels_025))), colors):
    Xk = X[labels_025 == k]
    ax3.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3, marker='.')
ax3.plot(X[labels_025 == -1, 0], X[labels_025 == -1, 1], 'k+', alpha=0.1)
ax3.set_title('Clustering at 0.25 epsilon cut\nDBSCAN')

# DBSCAN at 0.5
for k, c in zip(range(0, len(set(labels_05))), colors):
    Xk = X[labels_05 == k]
    ax5.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3)
ax5.plot(X[labels_05 == -1, 0], X[labels_05 == -1, 1], 'k+', alpha=0.1)
ax5.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

# DBSCAN at 0.75
for k, c in zip(range(0, len(set(labels_075))), colors):
    Xk = X[labels_075 == k]
    ax4.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3)
ax4.plot(X[labels_075 == -1, 0], X[labels_075 == -1, 1], 'k+', alpha=0.1)
ax4.set_title('Clustering at 0.75 epsilon cut\nDBSCAN')



plt.tight_layout()
plt.show()
