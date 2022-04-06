import math
import numpy as np
from utils.clustering.auxiliary_functions import distance

def approximationScheme(X):
    newX = np.zeros((len(X), len(X[0])))
    k = 0
    newX[k] = X[0]
    last = newX[k]
    for i in range(1, len(X)):
        if distance(X[i], last) > 2.5:
            k = k + 1
            newX[k] = X[i]
            last = newX[k]
    newX = newX[:k]
    print('Initial length: ' + str(len(X)))
    print('Rarefied length: ' + str(len(newX)))
    return newX