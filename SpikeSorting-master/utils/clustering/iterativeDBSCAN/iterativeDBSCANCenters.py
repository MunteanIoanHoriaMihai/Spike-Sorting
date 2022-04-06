import numpy as np
import pandas as pd
import math
import random

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from matplotlib import pyplot as plt

from utils.clustering.trim.approximation_scheme import approximationScheme
import utils.clustering.auxiliary_functions as fs
from utils.constants import LABEL_COLOR_MAP

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
X = approximationScheme(X)
initialX = X

clusters = np.full(len(X), -1)
nrClusters = 0
initialClusters = 0
currentNoise = lastNoise = len(X)
q=0
while True:
	lastNoise = len(X)
	
	min_samples=int(np.log(len(X))*10)
	
	NN = NearestNeighbors(n_neighbors=min_samples).fit(X)
	distances, indices = NN.kneighbors(X)
	fig = plt.figure()
	plt.plot(np.sort(distances[:, distances.shape[1]-1]), color='red')
	plt.show()
	print("Introduce eps: ")
	eps = float(input())
	
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
	plt.show()
	
	X = fs.keepNoise(X, labels)
	currentNoise = len(X)
	q+=1
	if (currentNoise == lastNoise):
		break


	

fig = plt.figure()
plt.scatter(initialX[:, 0], initialX[:, 1], marker='o', c=clusters, s=25, edgecolor='k')



"""		
from imblearn.under_sampling import RandomUnderSampler

sampler = RandomUnderSampler(random_state=0)
reducedX, reducedC = sampler.fit_resample(initialX, clusters)
                                     
	



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

clusterCenters = np.zeros((nrClusters, 2))
for c in range(0, nrClusters):
	k=0
	sumX=0
	sumY=0
	for i in range(0, len(initialX)):
		if clusters[i] == c:
			sumX +=initialX[i,0]
			sumY +=initialX[i,1]
			k += 1
	clusterCenters[c,0] = sumX/k
	clusterCenters[c,1] = sumY/k

for c in range(0, nrClusters):
	print(clusterCenters[c])
	
for c in range(initialClusters, nrClusters):
	print("#CLUSTER"+str(c))
	for i in range(0, initialClusters):
		print("#DISTANCE to "+str(c) + " is "+str(fs.distance(clusterCenters[i], clusterCenters[c])))
		 
"""
for testCluster in range(initialClusters, nrClusters):
	for i in range(0, initialClusters): # 0 - is noise, 1-2 main clusters
		



fig = plt.figure()
plt.scatter(reducedX[:, 0], reducedX[:, 1], marker='o', c=reducedClusters, s=25, edgecolor='k')

unique, counts = np.unique(finalClusters, return_counts=True)
print('#FINAL:'+ str(dict(zip(unique, counts))))

label_color = [LABEL_COLOR_MAP[l] for l in finalClusters]

fig = plt.figure()
plt.scatter(initialX[:, 0], initialX[:, 1], marker='o', c=label_color, s=25, edgecolor='k')
"""

plt.show() 

