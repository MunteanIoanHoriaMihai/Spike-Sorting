import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt

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


X = np.append(c1[1:25000], c3[1:25000], axis=1)
sortedX = np.sort(X, axis=0)

distribution = np.zeros(len(X))
k=0
for i in range(0, len(X)):
	distribution[k] += 1
	if not X[i-1,0]+0.75>X[i,0]:
		k += 1
distribution = distribution[:k]
print(k)

plt.plot(distribution, color='red')
plt.show() 