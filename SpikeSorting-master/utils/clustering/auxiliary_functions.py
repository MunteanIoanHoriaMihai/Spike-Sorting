import math
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state
import numbers
from sklearn.utils import shuffle as shuffle_


def densityTable(X, n):
    matrix = np.zeros((n, n), dtype=int)
    xStart = -5
    xEnd = 5
    yStart = 15
    yEnd = -10
    xIteration = (xEnd - xStart) / n
    yIteration = (yEnd - yStart) / n
    yCurrent = yStart
    i = 0
    print(xIteration)
    print(yIteration)
    while yCurrent > yEnd:
        j = 0
        xCurrent = xStart
        while xCurrent < xEnd:
            for k in range(0, len(X)):
                if xCurrent < X[k, 0] < xCurrent + xIteration and yCurrent + yIteration < X[k, 1] < yCurrent:
                    matrix[i, j] += 1
            xCurrent += xIteration
            j += 1
        yCurrent += yIteration
        i += 1
    return matrix


def neighbours(X, eps):
    neighbourMatrix = np.full((len(X), len(X)), int(-1))
    neighboursInEpsRadius = np.zeros(len(X))

    NN = NearestNeighbors(n_neighbors=len(X)).fit(X)
    distances, indices = NN.kneighbors(X)

    NN1 = NearestNeighbors(n_neighbors=int(np.log(len(X)))).fit(X)
    distances1, indices1 = NN1.kneighbors(X)

    for i in range(0, len(X)):
        k = 0
        for j in range(0, len(X)):
            if distances[i][j] < eps:
                neighboursInEpsRadius[i] = neighboursInEpsRadius[i] + 1
                neighbourMatrix[i][k] = j
                k = k + 1
    return neighbourMatrix, neighboursInEpsRadius, distances, distances1


def distance(pointA, pointB):
    sum = 0
    for i in range(0, len(pointA)):
        sum += (pointA[i] - pointB[i]) ** 2
    return math.sqrt(sum)


def applyDBSCAN(X, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('DBSCAN: Estimated number of clusters: %d' % n_clusters_)

    unique, counts = np.unique(labels, return_counts=True)
    print('DBSCAN:' + str(dict(zip(unique, counts))))

    return labels


def keepNoise(input, labels):
    output = np.zeros((len(input), 2))
    k = 0
    for i in range(0, len(input)):
        if labels[i] == -1:
            output[k] = input[i]
            k = k + 1
    output = output[:k]
    print('NOISE:' + str(len(output)))
    return output


def countOnes(array):
    k = 0
    for i in range(0, len(array)):
        if array[i] == 1:
            k += 1
    return k


def getIndice(list):
    for i in range(0, len(list)):
        if list[i] == 1:
            return i



def generate_overlapping_clusters(n_samples=100, centers=2, random_state=None):
    generator = check_random_state(random_state)

    centers = check_array(centers)
    n_features = centers.shape[1]

    X = []
    y = []

    n_centers = centers.shape[0]

    n_samples_per_center = [int(n_samples // n_centers)] * n_centers
    for i in range(n_samples % n_centers):
        n_samples_per_center[i] += 1

    for i, (n, std) in enumerate(zip(n_samples_per_center, np.ones(len(centers)))):
        X.append(centers[i] + generator.normal(scale=std, size=(n, n_features)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    return X, y


def make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None):
    """Generate isotropic Gaussian blobs for clustering.
    Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    ----------
    n_samples : int, or tuple, optional (default=100)
        The total number of points equally divided among clusters.
    n_features : int, optional (default=2)
        The number of features for each sample.
    centers : int or array of shape [n_centers, n_features], optional
        (default=3)
        The number of centers to generate, or the fixed center locations.
    cluster_std: float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.
    center_box: pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center when centers are
        generated at random.
    shuffle : boolean, optional (default=True)
        Shuffle the samples.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    Examples
    --------
    from sklearn.datasets.samples_generator import make_blobs
    X, y = make_blobs(n_samples=10, centers=3, n_features=2,
    ...                   random_state=0)
    print(X.shape)
    (10, 2)
    y
    array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])
    See also
    --------
    make_classification: a more intricate variant
    """
    generator = check_random_state(random_state)

    if isinstance(centers, numbers.Integral):
        centers = generator.uniform(center_box[0], center_box[1],
                                    size=(centers, n_features))
    else:
        centers = check_array(centers)
        n_features = centers.shape[1]

    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.ones(len(centers)) * cluster_std

    X = []
    y = []

    n_centers = centers.shape[0]
    if isinstance(n_samples, numbers.Integral):
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers
        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1
    else:
        n_samples_per_center = n_samples

    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(centers[i] + generator.normal(scale=std, size=(n, n_features)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    if shuffle:
        X, y = shuffle_(X, y, random_state=generator)

    return X, y
