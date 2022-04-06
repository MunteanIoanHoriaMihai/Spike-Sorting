import numpy as np
import pandas as pd
import math
import random

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

import utils.clustering.auxiliary_functions as fs

# Importing the dataset
data = pd.read_csv('../../../datasets/data.csv', skiprows=0)
f1 = data['F1'].values
f2 = data['F2'].values
f3 = data['F3'].values

#print("Shape:")
#print(data.shape)
#print("\n")

c1 = np.array([f1]).T
c2 = np.array([f2]).T
c3 = np.array([f3]).T


X = np.append(c1, c3, axis=1)
initialX = X

eps = 0.2
min_samples=int(np.log(len(X))*10)

NN = NearestNeighbors(n_neighbors=min_samples).fit(X)
distances, indices = NN.kneighbors(X)
fig = plt.figure()
plt.plot(np.sort(distances[:, distances.shape[1]-1]), color='red')

clusters = np.full(len(X), -1)
nrClusters = 0
initialClusters = 0
for q in range(0, 5):
	labels = fs.applyDBSCAN(X, eps, min_samples)
	
	if q == 0:
		clusters = labels
		nrClusters = np.amax(clusters) + 1
		initialClusters = nrClusters
	else: 
		k=0
		for i in range(0, len(clusters)):
			if clusters[i] == -1:
				if not labels[k] == -1:
					clusters[i] = labels[k] + nrClusters
				k+=1
				
	unique, counts = np.unique(clusters, return_counts=True)
	print('#CLUSTER:'+ str(dict(zip(unique, counts))))
	nrClusters = np.amax(clusters) + 1
	print('#CLUSTER:'+ str(nrClusters))
					
	
		
	
	fig = plt.figure()
	plt.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')
	X = fs.keepNoise(X, labels)
	
	
	eps += 0.2
	min_samples=int(np.log(len(X))*10)

fig = plt.figure()
plt.scatter(initialX[:, 0], initialX[:, 1], marker='o', c=clusters, s=25, edgecolor='k')




from imblearn.under_sampling import RandomUnderSampler

sampler = RandomUnderSampler(random_state=0)
reducedX, reducedC = sampler.fit_resample(initialX, clusters)
                                     
"""			



k=0
reducedX = np.zeros((len(initialX),2))
reducedC = np.zeros(len(initialX))
for c in range(0, nrClusters+1):
	for i in range(0, len(initialX)):
		if clusters[i] == c-1 and random.random()<0.1:
			reducedX[k]=initialX[i]
			reducedC[k]=clusters[i]
			k+=1
reducedX = reducedX[:k]
reducedC = reducedC[:k]
print(len(initialX))
print(len(reducedX))
print(len(reducedC))

unique, counts = np.unique(reducedC, return_counts=True)
print('#REDUCED:'+ str(dict(zip(unique, counts))))
"""

clusterIndices = np.full((nrClusters+1, len(reducedX)+1), -1)
for i in range(0, nrClusters+1):
	k=0
	for j in range(0, len(reducedX)):
		if reducedC[j] == i-1:
			clusterIndices[i, k] = j
			k+=1

finalClusters = reducedC

for testCluster in range(3, nrClusters+1):
	for i in range(0, 2): # 0 - is noise, 1-2 main clusters
		print(testCluster)
		flag=0
		for j in range(0, len(reducedX)): #j - indice for main cluster
			if clusterIndices[i, j] == -1:
				break
			for l in range(0, len(reducedX)): #l - indice for secondary cluster
				if math.sqrt((reducedX[clusterIndices[testCluster, l]][0] - reducedX[clusterIndices[i, j]][0])**2 + 
							 (reducedX[clusterIndices[testCluster, l]][1] - reducedX[clusterIndices[i, j]][1])**2) < 0.2:
					print("MUIE")
					flag=1
					break
			if flag == 1:
				break
		if flag == 1:
			for h in range(0, len(reducedX)):
				if reducedC[h] == testCluster:
					finalClusters[h] = 1




fig = plt.figure()
plt.scatter(reducedX[:, 0], reducedX[:, 1], marker='o', c=finalClusters, s=25, edgecolor='k')

unique, counts = np.unique(finalClusters, return_counts=True)
print('#FINAL:'+ str(dict(zip(unique, counts))))

plt.show() 

