from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.dataset_parsing.clustering_datasets import getTINSData
from approximation_scheme import approximationScheme
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


X, _ = getTINSData()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:,0] , X[:,1], X[:,2])

newX = approximationScheme(X)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(newX[:,0] , newX[:,1], newX[:,2])
plt.show()
