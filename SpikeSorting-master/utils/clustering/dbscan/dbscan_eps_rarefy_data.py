import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import random

from utils.clustering.auxiliary_functions import generate_overlapping_clusters, neighbours

print("------------- Initial data: -------------")
fig, ((plt1, plt2, plt3)) = plt.subplots(1, 3, figsize=(16, 4))
fig.tight_layout()

n_samples = 10000
random_state = 170
centers = [(-9, -6), (-3, -3), (1, 0)]
X, y = generate_overlapping_clusters(n_samples=n_samples, centers=centers, random_state=random_state)


for i in range(0, y.size):
	if y[i]==0:
		transformation = [[0.5, -0.25], [-0.5, 1]]
		X[i] = np.dot(X[i], transformation)	
	if y[i]==1:
		transformation = [[0.5, 0], [-0.5, 1]]
		X[i] = np.dot(X[i], transformation)	

plt1.set_xlim(-5, 5)
plt1.set_ylim(-10, 5)
plt1.set_title('Original data')
plt1.scatter(X[:, 0], X[:, 1], marker='o', s=25, edgecolor='k')

#----------------- PLT4 - ELBOW ----------------------
# for no undersampling
NN1 = NearestNeighbors(n_neighbors=np.log(X.size).astype(int)).fit(X)
distances1, indices1 = NN1.kneighbors(X)

plt2.set_title('eps elbow')
plt2.plot(np.sort(distances1[:, distances1.shape[1]-1]), color='red', label = 'NoUndersample')
plt2.legend()

print(X.shape)
print(len(X))

### DBSCAN
eps=0.26
min_samples=np.log(n_samples)
#min_samples=7
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))
plt3.set_title('DBSCAN on OG DATA')
plt3.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

NN = NearestNeighbors(n_neighbors=len(X)).fit(X)
distances, indices = NN.kneighbors(X)




# Eps Distance Nearest Neighbour
# epsNeighbours[i] - nr of neighbours of point x, where distance is smaller than eps
def calculate_eps_neighbours(X, eps):
	epsNeighbours = np.zeros(len(X))
	for i in range(0, len(X)-1):
		for j in range(0, len(X)-1):
			if distances[i][j]<eps:
				epsNeighbours[i] = epsNeighbours[i] +1
	return epsNeighbours

###################################################################################
###################################################################################
###################################################################################
print("\n------------- Limit by number of neighbours: -------------")
fig, ((plt1, plt2, plt3, plt4)) = plt.subplots(1, 4, figsize=(16, 4))
fig.tight_layout()


eps=0.26
epsNeighbours = calculate_eps_neighbours(X, eps)
plt1.set_title(f'eps={eps} distance neighbours')
plt1.plot(epsNeighbours, color='red')

sortedEpsNeighbours = np.sort(epsNeighbours)
plt2.set_title('eps distance neighbours sorted')
plt2.plot(sortedEpsNeighbours, color='red')

# Throw out extremes, very high and low densities
# sortedEpsNeighbours[len(X)-1] the max number of neighbours any point has  
# newX = the new X in which there are points that have only between 20% and 80% number of neighbours 
# k - counter for new data
newX = np.zeros((len(X),2))
k=0
for i in range(0, len(X)-1):
	if 0.2<epsNeighbours[i]/sortedEpsNeighbours[len(X)-1]<0.8:
		newX[k]=X[i]
		k=k+1
newX = newX[:k]

plt3.set_title('Rarefied data - \nonly points with 20-80% of neighbours')
plt3.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')


#distanceMatrix[i][j] - distance from point i to j 
#distanceMatrix = np.zeros((k,k))
#for i in range(0, k-1):
#	for j in range(0, k-1):
#		distanceMatrix[i][j]= math.sqrt((X[i,0]-X[j,0])**2+(X[i,1]-X[j,1])**2)
NNnew = NearestNeighbors(n_neighbors=k).fit(newX)
distanceMatrix, indices = NNnew.kneighbors(X)

#finalX - final data 
# p - counter for final data length
finalX = np.zeros((k,2))
p=0
for i in range(0, k-1):
	#count is the number of points in radius eps of point i
	count = 0
	for j in range(0, k-1):
		if distanceMatrix[i][j] < eps:
			count = count+1
	
	if 3<count<30:
		finalX[p] = newX[i]
		p = p+1
finalX = finalX[:p]
print(len(finalX))


plt4.set_title('Final data - \nwith 3-30 neighbours<eps from rarefied ')
plt4.scatter(finalX[:, 0], finalX[:, 1], marker='o', s=25, edgecolor='k')


###################################################################################
###################################################################################
###################################################################################
fig, ((plt1, plt2, plt3, plt4)) = plt.subplots(1, 4, figsize=(16, 4))
fig.tight_layout()

eps=0.1
epsNeighbours = calculate_eps_neighbours(X, eps)
plt1.set_title(f'eps={eps} distance neighbours')
plt1.plot(epsNeighbours, color='red')

sortedEpsNeighbours = np.sort(epsNeighbours)
plt2.set_title('eps distance neighbours sorted')
plt2.plot(sortedEpsNeighbours, color='red')

newX = np.zeros((len(X), 2))
flag = np.ones(len(X))
k = 0
for i in range(0, len(X) - 1):
	# take only high density (80% or more from max density)
	if epsNeighbours[i] / sortedEpsNeighbours[len(X) - 1] > 0.8:
		for j in range(0, len(X) - 1):
			# keep only points that arent close
			if distances[i][j] / eps < 0.5:
				flag[j] = 0

flagX = np.array(flag, dtype=np.bool)
X = np.array(X)
newX = X[flagX]
print(len(newX))
plt3.set_title('Rarefied data - \n only >80% neighbours')
plt3.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')

# ----------------- DBSCAN ----------------------
min_samples = np.log(n_samples)
# min_samples=7
db = DBSCAN(eps=eps, min_samples=min_samples).fit(newX)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))
plt4.set_title('DBSCAN on rarefied data')
plt4.scatter(newX[:, 0], newX[:, 1], marker='o', c=labels, s=25, edgecolor='k')


###################################################################################
###################################################################################
###################################################################################
fig, ((plt1, plt2, plt3)) = plt.subplots(1, 4, figsize=(16, 4))
fig.tight_layout()

#eps=3
eps=0.2
# eps=0.1
epsNeighbours = calculate_eps_neighbours(X, eps)
plt1.set_title(f'eps={eps} distance neighbours')
plt1.plot(epsNeighbours, color='red')

sortedEpsNeighbours = np.sort(epsNeighbours)
plt2.set_title('eps distance neighbours sorted')
plt2.plot(sortedEpsNeighbours, color='red')

valoarea = int(sortedEpsNeighbours[len(X)-1]/20)
print(valoarea)
drift = int(len(X)/100)
print(drift)
index=0
for i in range(0+drift, len(X)-drift):
	if (sortedEpsNeighbours[i-drift]+5)<sortedEpsNeighbours[i] and (sortedEpsNeighbours[i+drift]-5)>sortedEpsNeighbours[i]:
		index=i
		print(index)
		break

densityVector = np.zeros((len(X), 2))
for i in range(0, len(X)):
	densityVector[i] = [i, epsNeighbours[i]]

sortedDensityVector = densityVector[densityVector[:, 1].argsort()[::-1]]
print(sortedDensityVector[0, 1])

flagX = np.ones(len(X))
for i in range(0, len(X)):
	for j in range(0, len(X)):
		if distances[int(sortedDensityVector[i, 0]), j] < eps:
			flagX[j] = 0


flagX = np.array(flagX, dtype=np.bool)
X = np.array(X)
newX = X[flagX]
print(len(newX))

plt3.set_title('Rarefied data')
plt3.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')


###################################################################################
###################################################################################
###################################################################################
fig, ((plt1, plt2, plt3, plt4, plt5, plt6)) = plt.subplots(1, 6, figsize=(16, 4))
fig.tight_layout()

#eps=3
eps=0.2
# Eps Distance Nearest Neighbour
# neighboursInEpsRadius[i] - nr of neighbours of point x, where distance is smaller than eps

neighbourMatrix, neighboursInEpsRadius, distances, distances1 = neighbours(X, eps)

plt1.set_title('eps distance neighbours')
plt1.plot(neighboursInEpsRadius, color='red')

sortedEpsNeighbours = np.sort(neighboursInEpsRadius)
plt2.set_title('eps distance neighbours sorted')
plt2.plot(sortedEpsNeighbours, color='red')

densityVector = np.zeros((len(X), 2))
for i in range(0, len(X)):
	densityVector[i] = [i, neighboursInEpsRadius[i]]

sortedDensityVector = densityVector[densityVector[:, 1].argsort()[::-1]]

#------ HIGH DENSITY -------
highDensity = np.zeros((5000, 2))
for i in range(0, len(highDensity)):
	highDensity[i] = X[int(sortedDensityVector[i, 0])]

plt3.set_xlim(-5, 5)
plt3.set_ylim(-10, 5)
plt3.set_title('High Density data')
plt3.scatter(highDensity[:, 0], highDensity[:, 1], marker='o', s=25, edgecolor='k')

# ------ RAREFICATION -------
k = 0
densityThreshold = sortedDensityVector[5000, 1]
flagX = np.ones(len(X))
for i in range(0, len(X)):
	if flagX[int(sortedDensityVector[i, 0])] == 1:
		flagX[int(sortedDensityVector[i, 0])] = 2
	if int(sortedDensityVector[i, 1]) > densityThreshold:
		for j in range(0, len(X)):
			if neighbourMatrix[int(sortedDensityVector[i, 0])][j] == -1:
				break
			if flagX[neighbourMatrix[int(sortedDensityVector[i, 0])][j]] == 1:
				flagX[neighbourMatrix[int(sortedDensityVector[i, 0])][j]] = 0

flagX = np.array(flagX, dtype=np.bool)
X = np.array(X)
newX = X[flagX]

print(len(newX))

plt4.set_xlim(-5, 5)
plt4.set_ylim(-10, 5)
plt4.set_title('Rarefied data - remove high density')
plt4.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')

###################################################################################
###################################################################################
###################################################################################
fig, ((plt1, plt2, plt3, plt4)) = plt.subplots(1, 4, figsize=(16, 4))
fig.tight_layout()

eps=0.2
# Eps Distance Nearest Neighbour
# epsNeighbours[i] - nr of neighbours of point x, where distance is smaller than eps
epsNeighbours = np.zeros(len(X))
for i in range(0, len(X)):
	for j in range(0, len(X)):
		if distances[i][j]<eps:
			epsNeighbours[i] = epsNeighbours[i] +1

plt1.set_title('eps distance neighbours')
plt1.plot(epsNeighbours, color='red')


sortedEpsNeighbours = np.sort(epsNeighbours)
#print(sortedEpsNeighbours[len(sortedEpsNeighbours)-1])
plt2.set_title('eps distance neighbours sorted')
plt2.plot(sortedEpsNeighbours, color='red')

min_samples=np.log(n_samples)
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt3.set_xlim(-5, 5)
plt3.set_ylim(-10, 5)
plt3.set_title(f'DBSCAN eps={eps}, min={min_samples}')
plt3.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

newX = np.zeros((len(X), 2))
k = 0
heighestNeighbour = 0
averageDensity = 0

averageOutlierDensity = 0
q = 0
for i in range(0, len(X)):
	if labels[i] == -1:
		averageOutlierDensity += epsNeighbours[i]
		q = q + 1
print(q)
print(averageOutlierDensity / q)
averageOutlierDensity /= q

for index, number in zip(unique, counts):
	if number > n_samples / 20 and not index == -1:
		heighestNeighbour = 0
		averageDensity = 0
		q = 0
		for i in range(0, len(X)):
			if labels[i] == index:
				if epsNeighbours[i] > heighestNeighbour:
					heighestNeighbour = epsNeighbours[i]
				averageDensity += epsNeighbours[i]
				q = q + 1
		print(index, heighestNeighbour, averageDensity / q)
		averageDensity /= q
		for i in range(0, len(X)):
			if labels[i] == index and random.random() < averageOutlierDensity / averageDensity:
				newX[k] = X[i]
				k = k + 1
newX = newX[:k]
print(len(newX))

for index, number in zip(unique, counts):
	if number <= n_samples / 20 or index == -1:
		for i in range(0, len(X)):
			if labels[i] == index:
				newX = np.vstack((newX, X[i]))
print(len(newX))

plt4.set_xlim(-5, 5)
plt4.set_ylim(-10, 5)
plt4.set_title('Rarefied')
plt4.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')


NN2 = NearestNeighbors(n_neighbors=np.log(newX.size).astype(int)).fit(newX)
distances2, indices2 = NN2.kneighbors(newX)

plt5.set_title('eps elbow rarefied')
plt5.plot(np.sort(distances2[:, distances2.shape[1]-1]), color='red', label = 'NoUndersample')
plt5.legend()

eps=0.2

min_samples=np.log(len(newX))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(newX)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt6.set_xlim(-5, 5)
plt6.set_ylim(-10, 5)
plt6.set_title('DBSCAN on Rarefied')
plt6.scatter(newX[:, 0], newX[:, 1], marker='o', c=labels, s=25, edgecolor='k')

###################################################################################
###################################################################################
###################################################################################
fig, ((plt1, plt2, plt3, plt4)) = plt.subplots(1, 4, figsize=(16, 4))
fig.tight_layout()

eps = 0.2
# Eps Distance Nearest Neighbour
# neighboursInEpsRadius[i] - nr of neighbours of point x, where distance is smaller than eps
neighbourMatrix, neighboursInEpsRadius, distances, distances1 = neighbours(X, eps)

plt1.set_title('eps distance neighbours')
plt1.plot(neighboursInEpsRadius, color='red')

sortedEpsNeighbours = np.sort(neighboursInEpsRadius)
valoarea = int(sortedEpsNeighbours[len(X) - 1] / 4 - 25)
print(valoarea)
drift = int(len(X) / 10)
print(drift)
index = 0
for i in range(0 + drift, len(X) - drift):
	if (sortedEpsNeighbours[i - drift] + valoarea) < sortedEpsNeighbours[i] and (
			sortedEpsNeighbours[i + drift] - valoarea) > sortedEpsNeighbours[i]:
		index = i
		print(index)
		break
plt2.set_title('eps distance neighbours sorted')
plt2.plot(sortedEpsNeighbours, color='red')

densityVector = np.zeros((len(X), 2))
for i in range(0, len(X)):
	densityVector[i] = [i, neighboursInEpsRadius[i]]

sortedDensityVector = densityVector[densityVector[:, 1].argsort()[::-1]]

# ------ HIGH DENSITY -------
highDensity = np.zeros((len(X)-index, 2))
for i in range(0, len(highDensity)):
	highDensity[i] = X[int(sortedDensityVector[i, 0])]

plt3.set_xlim(-5, 5)
plt3.set_ylim(-10, 5)
plt3.set_title('High Density data')
plt3.scatter(highDensity[:, 0], highDensity[:, 1], marker='o', s=25, edgecolor='k')

# ------ RAREFICATION -------

densityThreshold = sortedDensityVector[len(highDensity), 1]
flagX = np.ones(len(X))
for i in range(0, len(X)):
	if flagX[int(sortedDensityVector[i, 0])] == 1:
		flagX[int(sortedDensityVector[i, 0])] = 2
	if int(sortedDensityVector[i, 1]) > densityThreshold:
		for j in range(0, len(X)):
			if neighbourMatrix[i][j] == -1:
				break
			if flagX[j] == 1:
				flagX[j] = 0

flagX = np.array(flagX, dtype=np.bool)
X = np.array(X)
newX = X[flagX]

print(len(newX))

plt4.set_title('Rarefied data')
plt4.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')

###################################################################################
###################################################################################
###################################################################################
print("\n------------- Remove outliers - by neighbour number: -------------")
fig, ((plt1, plt2, plt3, plt4)) = plt.subplots(1, 4, figsize=(16, 4))
fig.tight_layout()

eps=0.2
neighbourMatrix, neighboursInEpsRadius, distances, distances1 = neighbours(X, eps)

plt2.set_title('eps elbow')
plt2.plot(np.sort(distances1[:, distances1.shape[1]-1]), color='red', label = 'NoUndersample')
plt2.legend()

sortedEpsNeighbours = np.sort(neighboursInEpsRadius)
plt1.set_title('eps distance neighbours sorted')
plt1.plot(sortedEpsNeighbours, color='red')

densityVector = np.zeros((len(X), 2))
for i in range(0, len(X)):
	densityVector[i] = [i, neighboursInEpsRadius[i]]

sortedDescDensityVector = densityVector[densityVector[:, 1].argsort()[::-1]]
sortedDensityVector = densityVector[densityVector[:, 1].argsort()]

print(sortedDensityVector[1000, 1])

# min_samples=np.log(n_samples)
min_samples = np.log(n_samples) * 10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt3.set_xlim(-5, 5)
plt3.set_ylim(-10, 5)
plt3.set_title('DBSCAN')
plt3.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')



# ------ RAREFICATION -------
newX = np.zeros((len(X),2))
k=0
for i in range(0, len(X)):
	if neighboursInEpsRadius[i]>neighboursInEpsRadius[1000]:
		newX[k]=X[i]
		k=k+1
newX = newX[:k]
print(len(newX))


min_samples=np.log(len(newX))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(newX)
labels = db.labels_


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt4.set_xlim(-5, 5)
plt4.set_ylim(-10, 5)
plt4.set_title('Rarefied data')
plt4.scatter(newX[:, 0], newX[:, 1], c=labels, marker='o', s=25, edgecolor='k')

###################################################################################
###################################################################################
###################################################################################
print("\n------------- Rarefy - select by DBSCAN: -------------")
fig, ((plt1, plt2, plt3, plt4, plt5, plt6)) = plt.subplots(1, 6, figsize=(16, 4))
fig.tight_layout()


eps=0.25
NN = NearestNeighbors(n_neighbors=len(X)).fit(X)
distances, indices = NN.kneighbors(X)

# Eps Distance Nearest Neighbour
# epsNeighbours[i] - nr of neighbours of point x, where distance is smaller than eps
epsNeighbours = np.zeros(len(X))
for i in range(0, len(X)):
	for j in range(0, len(X)):
		if distances[i][j]<eps:
			epsNeighbours[i] = epsNeighbours[i] +1


plt1.set_title('eps distance neighbours')
plt1.plot(epsNeighbours, color='red')


sortedEpsNeighbours = np.sort(epsNeighbours)
#print(sortedEpsNeighbours[len(sortedEpsNeighbours)-1])
plt2.set_title('eps distance neighbours sorted')
plt2.plot(sortedEpsNeighbours, color='red')

# min_samples=np.log(n_samples)
min_samples=np.log(len(X))*10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt3.set_xlim(-5, 5)
plt3.set_ylim(-10, 5)
plt3.set_title('DBSCAN')
plt3.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

newX = np.zeros((len(X),2))
k=0
for index, number in zip(unique, counts):
	if not index==-1:
		for i in range(0, len(X)):
			if labels[i]==index:
				newX[k]=X[i]
				k=k+1
newX = newX[:k]
print(len(newX))


plt4.set_xlim(-5, 5)
plt4.set_ylim(-10, 5)
plt4.set_title('Rarefied')
plt4.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')


NN2 = NearestNeighbors(n_neighbors=np.log(newX.size).astype(int)).fit(newX)
distances2, indices2 = NN2.kneighbors(newX)

plt5.set_title('eps elbow')
plt5.plot(np.sort(distances2[:, distances2.shape[1]-1]), color='red', label = 'NoUndersample')
plt5.legend()

eps=0.25

min_samples=np.log(len(newX))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(newX)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt6.set_xlim(-5, 5)
plt6.set_ylim(-10, 5)
plt6.set_title('DBSCAN')
plt6.scatter(newX[:, 0], newX[:, 1], marker='o', c=labels, s=25, edgecolor='k')


###################################################################################
###################################################################################
###################################################################################
print("\n------------- Rarefy - select by DBSCAN: -------------")
fig, ((plt1, plt2, plt3, plt4, plt5, plt6)) = plt.subplots(1, 6, figsize=(16, 4))
fig.tight_layout()

eps=0.25
NN = NearestNeighbors(n_neighbors=len(X)).fit(X)
distances, indices = NN.kneighbors(X)


# Eps Distance Nearest Neighbour
# epsNeighbours[i] - nr of neighbours of point x, where distance is smaller than eps
epsNeighbours = np.zeros(len(X))
for i in range(0, len(X)):
	for j in range(0, len(X)):
		if distances[i][j]<eps:
			epsNeighbours[i] = epsNeighbours[i] +1

plt1.set_title('eps distance neighbours')
plt1.plot(epsNeighbours, color='red')


sortedEpsNeighbours = np.sort(epsNeighbours)
#print(sortedEpsNeighbours[len(sortedEpsNeighbours)-1])
plt2.set_title('eps distance neighbours sorted')
plt2.plot(sortedEpsNeighbours, color='red')

min_samples = np.log(n_samples)
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt3.set_xlim(-5, 5)
plt3.set_ylim(-10, 5)
plt3.set_title('DBSCAN')
plt3.scatter(X[:, 0], X[:, 1], marker='o', c=labels, s=25, edgecolor='k')

newX = np.zeros((len(X), 2))
k = 0
heighestNeighbour = 0
averageDensity = 0

averageOutlierDensity = 0
q = 0
for i in range(0, len(X)):
	if labels[i] == -1:
		averageOutlierDensity += epsNeighbours[i]
		q = q + 1
print(q)
print(averageOutlierDensity / q)
averageOutlierDensity /= q

for index, number in zip(unique, counts):
	if number > n_samples / 20 and not index == -1:
		heighestNeighbour = 0
		averageDensity = 0
		q = 0
		for i in range(0, len(X)):
			if labels[i] == index:
				if epsNeighbours[i] > heighestNeighbour:
					heighestNeighbour = epsNeighbours[i]
				averageDensity += epsNeighbours[i]
				q = q + 1
		print(index, heighestNeighbour, averageDensity / q)
		averageDensity /= q
		for i in range(0, len(X)):
			if labels[i] == index and random.random() > averageOutlierDensity / averageDensity:
				newX[k] = X[i]
				k = k + 1
newX = newX[:k]
print(len(newX))

plt4.set_xlim(-5, 5)
plt4.set_ylim(-10, 5)
plt4.set_title('Rarefied')
plt4.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')

NN2 = NearestNeighbors(n_neighbors=np.log(newX.size).astype(int)).fit(newX)
distances2, indices2 = NN2.kneighbors(newX)

plt5.set_title('eps elbow')
plt5.plot(np.sort(distances2[:, distances2.shape[1] - 1]), color='red', label='NoUndersample')
plt5.legend()

eps = 0.25

min_samples = np.log(len(newX))
db = DBSCAN(eps=eps, min_samples=min_samples).fit(newX)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

plt6.set_xlim(-5, 5)
plt6.set_ylim(-10, 5)
plt6.set_title('DBSCAN')
plt6.scatter(newX[:, 0], newX[:, 1], marker='o', c=labels, s=25, edgecolor='k')



###################################################################################
###################################################################################
###################################################################################
print("\n------------- Rarefy: -------------")
fig, ((plt1, plt2)) = plt.subplots(1, 2, figsize=(16, 4))
fig.tight_layout()

eps=0.20
NN = NearestNeighbors(n_neighbors=len(X)).fit(X)
distances, indices = NN.kneighbors(X)

# Eps Distance Nearest Neighbour
# epsNeighbours[i] - nr of neighbours of point x, where distance is smaller than eps
epsNeighbours = np.zeros(len(X))
flagX = np.ones(len(X))
for i in range(0, len(X)-1):
	for j in range(0, len(X)-1):
		if distances[i][j]<eps:
			epsNeighbours[i] = epsNeighbours[i] +1
			flagX[j] = 0
plt1.set_title('eps distance neighbours')
plt1.plot(epsNeighbours, color='red')


sortedEpsNeighbours = np.sort(epsNeighbours)
plt3.set_title('eps distance neighbours sorted')
plt3.plot(sortedEpsNeighbours, color='red')

flagX = np.array(flagX, dtype=np.bool)
X = np.array(X)
newX = X[flagX]

print(len(newX))

plt2.set_title('Rarefied data')
plt2.scatter(newX[:, 0], newX[:, 1], marker='o', s=25, edgecolor='k')



plt.show()


