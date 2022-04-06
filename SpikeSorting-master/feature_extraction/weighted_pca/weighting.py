import numpy as np


def apply_weights(X, peaks, div=None):
    """
    Modifies the feature space by multiplying the features with a weight composed from the number of peaks and
    divergence, if it is different from None.

    @param X: 2D array of shape (n_samples, n_features), stores the data
    @param peaks: 1D array, the number of peaks for each data feature
    @param div: 1D array, divergence coefficient for each data feature
    @return: Y - 2D array, modified dataset
             weights - 1D array, the weight applied for each data feature
    """

    reference = np.max(peaks)

    Y = np.zeros((X.shape[0], X.shape[1]))
    weights = []

    for i in range(len(peaks)):
        w = peaks[i] / reference

        if div is not None:
            w *= 10*div[i]

        weights.append(w)

        Y[:, i] = X[:, i] * w

    return Y, weights