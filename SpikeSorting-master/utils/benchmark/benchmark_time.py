import sys

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

sys.setrecursionlimit(100000)

from utils.sbm import SBM
from utils.dataset_parsing import simulations_dataset as ds
#
# dataName = ["S1", "S2", "U", "UO"]
# files = ["s1_labeled.csv", "s2_labeled.csv", "unbalance.csv"]
# kmeansValues = [15, 15, 8, 6]
# epsValues = [27000, 45000, 18000, 0.5]
# pn = 25
# numberOfIterations = 10
# for i in range(0, 4):
#     if i<3:
#         X = np.genfromtxt("./datasets/"+files[i], delimiter=",")
#         X, y = X[:, [0, 1]], X[:, 2]
#     else:
#         X, y = ds.getGenData()
#
#     sum = 0
#     for j in range(0, numberOfIterations):
#         start = timer()
#         kmeans = KMeans(n_clusters=kmeansValues[i]).fit(X)
#
#         labels = kmeans.labels_
#         end = timer()
#         sum = sum + (end - start)
#     print(dataName[i] + " - KMEANS TIME: "+str(sum/numberOfIterations))
#
#
#
#     sum = 0
#     min_samples = np.log(len(X))
#     for j in range(0, numberOfIterations):
#         start = timer()
#         db = DBSCAN(eps=epsValues[i], min_samples=min_samples).fit(X)
#         labels = db.labels_
#         end = timer()
#         sum = sum + (end - start)
#     print(dataName[i] + " - DBSCAN TIME: "+str(sum/numberOfIterations))
#
#     sum = 0
#     for j in range(0, numberOfIterations):
#         start = timer()
#         labels = SBM.sequential(X, pn)
#         end = timer()
#         sum = sum + (end - start)
#     print(dataName[i] + " - SBM TIME: "+str(sum/numberOfIterations))

import time


def get_simulation_average_time():
    averageKMeansTime = 0
    averageDBSCANTime = 0
    averageSBMv2Time = 0
    averageSBMv1Time = 0
    header = "Dataset Number, KMEANS, DBSCAN, SBM-V2, SBM-V1"
    allTimes = np.empty((5,))
    for i in range(1, 96):
        print(i)
        if i == 24 or i == 25 or i == 44:
            continue
        X, y = ds.get_dataset_simulation_pca_3d(simNr=i)

        kmeansTime = 0
        for j in range(0, 5):
            start = time.time()
            kmeans = KMeans(n_clusters=np.amax(y)).fit(X)
            labels = kmeans.labels_
            end = time.time()
            kmeansTime += (end - start)
        kmeansTime /= 5
        averageKMeansTime += kmeansTime

        dbscanTime = 0
        min_samples = np.log(len(X))
        for j in range(0, 5):
            start = time.time()
            db = DBSCAN(eps=0.5, min_samples=min_samples).fit(X)
            labels = db.labels_
            end = time.time()
            dbscanTime += (end - start)
        dbscanTime /= 5
        averageDBSCANTime += dbscanTime

        sbmv2Time = 0
        for j in range(0, 5):
            start = time.time()
            labels = SBM.sequential(X, pn=30, version=2)
            end = time.time()
            sbmv2Time += (end - start)
        sbmv2Time /= 5
        averageSBMv2Time += sbmv2Time

        sbmv1Time = 0
        for j in range(0, 5):
            start = time.time()
            labels = SBM.sequential(X, pn=30, version=1)
            end = time.time()
            sbmv1Time += (end - start)
        sbmv1Time /= 5
        averageSBMv1Time += sbmv1Time

        allTimes = np.vstack((allTimes, [i, kmeansTime, dbscanTime, sbmv2Time, sbmv1Time]))
    np.savetxt("PCA3D_time.csv", allTimes, delimiter=',', header=header, fmt="%10.2f")
    print("Average KMeans Time: {}".format(np.array(averageKMeansTime) / 92))
    print("Average DBSCAN Time: {}".format(np.array(averageDBSCANTime) / 92))
    print("Average SBM Time: {}".format(np.array(averageSBMv2Time) / 92))
    print("Average SBM Time: {}".format(np.array(averageSBMv1Time) / 92))


get_simulation_average_time()
