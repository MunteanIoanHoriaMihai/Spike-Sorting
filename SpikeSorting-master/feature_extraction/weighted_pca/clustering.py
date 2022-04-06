from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from utils.sbm import SBM
import numpy as np


def apply_clustering_algorithm(X, y, algorithm_number):
    """
    Applies clustering algorithm on data stored in X

    @param X: 2D array, stores the data
    @param y: 1D array, labels
    @param algorithm_number: integer, algorithm number
    @return: 1D array, clustering labels for each data sample of X
    """

    labels = []
    if algorithm_number == 0:
        kmeans = KMeans(n_clusters=int(np.amax(y)) + 1, n_init=10, init="k-means++", tol=1e-4, max_iter=300, verbose=0,
                        random_state=0).fit(X)
        labels = kmeans.labels_
        inertia = kmeans.inertia_
        n_iter = kmeans.n_iter_
        centroids = kmeans.cluster_centers_
    else:
        if algorithm_number == 1:
            labels = SBM.parallel(X, pn=10, ccThreshold=15, version=2)
        elif algorithm_number == 2:
            gmm = GaussianMixture(n_components=np.amax(y) + 1, random_state=0)
            labels = gmm.fit_predict(X)
        elif algorithm_number == 3:
            labels = AgglomerativeClustering(n_clusters=np.amax(y) + 1).fit_predict(X)
    return labels
