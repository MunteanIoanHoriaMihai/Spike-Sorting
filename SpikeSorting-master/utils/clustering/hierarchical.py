from copy import deepcopy
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


# Importing the dataset
data = pd.read_csv('../../datasets/data.csv', skiprows=0)
data = data.head(10000)
f1 = data['F1'].values
f2 = data['F2'].values
f3 = data['F3'].values

print("Shape:")
print(data.shape)
print("\n")

# create dendrogram
#dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
# create clusters
cluster = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage = 'ward')

# save clusters for chart
labels = cluster.fit_predict(data)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(f1,f2,f3, c=labels.astype(np.float))


plt.show() 
fig.savefig('hierarchical-2clusters-10000.jpg', dpi=100) 