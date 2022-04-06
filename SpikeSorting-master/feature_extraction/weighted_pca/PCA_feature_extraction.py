from sklearn.decomposition import PCA, KernelPCA
import numpy as np
import matplotlib.pyplot as plt


def compute_correlation_feature_pc(eigenvectors, eigenvalues, X):
    """
    Matrix of correlation coefficients between original features and the principal components

    @param eigenvectors:
    @param eigenvalues:
    @param X: 2D array, stores the data
    @return:
    """

    # compute the variance of each feature
    variances = []
    for i in range(0, X.shape[1]):
        variances.append(X[:, i].var())

    # compute loadings
    loadings = abs(eigenvectors.T) * np.sqrt(eigenvalues)

    # divide by standard deviation if not normalized
    loadings = loadings.T * np.sqrt(variances)

    return loadings.T


def subtract_feature_mean(X):
    """
    Computes the mean of each feature of dataset X and subtracts from it the mean of each feature.

    @param X: 2D array of shape (n_samples, n_features), stores the data
    @return: modified dataset X
    """

    means = []
    for i in range(0, 79):
        means.append(X[:, i].mean())

    for i in range(0, 79):
        X[:, i] = X[:, i] - means[i]

    return X


def cumulative_explained_variance(pca, nb_components):
    cum = np.cumsum(pca.explained_variance_)
    plt.plot(np.arange(nb_components), cum, color="orange")
    plt.xlabel("principal components")
    plt.ylabel('cumulative explained variance')
    plt.show()


def test_singular_values(features, X):
    """
    features = principal components = X reprojected on eigenvectors
    singular value = L2-norm of each component
    """

    sum = 0
    for i in range(0, X.shape[0]):
        sum += features[i, 1] * features[i, 1]

    np.sqrt(sum)


def apply_feature_extraction(X, method='pca', nb_components=None, print_info=False):
    """
    Applies feature extraction method on dataset X.

    @param X: 2D array of shape (n_samples, n_features), stores the data
    @param method: string, feature extraction method
    @param nb_components: integer, number of resulting features
    @param print_info: Boolean, True - prints the eigenvalue ratio of each principal component
    @return: 2D array (n_samples, nb_components), modified feature space
    """

    if method.lower() == 'pca':
        pca = PCA(n_components=nb_components, svd_solver='auto')
        principal_components = pca.fit_transform(X)

        eigenvectors_pca = pca.components_  # shape(n_components, n_features)
        eigenvalues_pca = pca.explained_variance_
        eigenvalues_ratio = pca.explained_variance_ratio_

        # correlation between each feature and principal component
        coefs = compute_correlation_feature_pc(eigenvectors_pca, eigenvalues_pca, X)

        # new_features = subtract_feature_mean(X)
        # ftrs = np.dot(new_features, eigenvectors_pca.T)
        features = principal_components
        # plot_heatmap(coefs, "", title)

        if print_info is True:
            print('Variance percentages')
            print(eigenvalues_ratio)

    elif method.lower() == 'kpca':
        kpca = KernelPCA(n_components=nb_components, kernel='linear')
        features = kpca.fit_transform(X)
    else:
        raise Exception('Dimensionality Reduction method unknown.')

    return features
