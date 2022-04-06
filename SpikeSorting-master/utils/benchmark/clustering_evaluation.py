from sklearn import metrics
import numpy as np
import utils.constants as cs


def external_clustering_validation(labels_true, labels_pred, no_noise_points=False):
    """
    External clustering validation with ARI and AMI.

    @param labels_true: 1D array, true clustering labels
    @param labels_pred: 1D array, resulted clustering labels
    @param no_noise_points: Boolean, True - consider the noise points, False - do not consider the noise points
    @return: 1D array, resulting scores
    """

    all_ari = np.round(metrics.adjusted_rand_score(labels_true, labels_pred), 3)
    all_ami = np.round(metrics.adjusted_mutual_info_score(labels_true, labels_pred), 3)

    if no_noise_points is True:
        adj = labels_pred > 0
        y_nn = labels_true[adj]
        labels_nn = labels_pred[adj]

        nnp_ari = np.round(metrics.adjusted_rand_score(y_nn, labels_nn), 3)
        nnp_ami = np.round(metrics.adjusted_mutual_info_score(y_nn, labels_nn), 3)

        return np.array([all_ari, all_ami, nnp_ari, nnp_ami])

    return np.array([all_ari, all_ami])


def print_external_clustering_scores(sim_nr, algorithm_number, score_values, no_noise_points=False):
    """
    Printing of external evaluation scores.

    @param sim_nr: integer, Simulation number
    @param algorithm_number: integer, algorithm number
    @param score_values: 1D array, scores' values
    @param no_noise_points: Boolean
    """

    print(
        "Simulation" + str(sim_nr) + " - " + cs.algorithms_[algorithm_number] + " - " + 'ARI: {: .7f}'.format(
            score_values[0]))
    print(
        "Simulation" + str(sim_nr) + " - " + cs.algorithms_[algorithm_number] + " - " + 'AMI: {: .7f}'.format(
            score_values[1]))

    if no_noise_points is True:
        print(
            "Simulation" + str(sim_nr) + " - " + cs.algorithms_[algorithm_number] + " - " + 'ARI-NNP: {: .3f}'.format(
                score_values[2]))
        print(
            "Simulation" + str(sim_nr) + " - " + cs.algorithms_[algorithm_number] + " - " + 'AMI-NNP: {: .3f}'.format(
                score_values[3]))


def internal_clustering_validation(x, labels):
    """
    Internal clustering validation with Silhouette, Calinski-Harabasz and Davies-Bouldin.

    @param x: 2D array of shape (n_samples, n_features), stores the data
    @param labels: 1D array (n_samples), clustering labels for each data sample
    @return: 1D array, resulting scores
    """

    s_score = np.round(metrics.silhouette_score(x, labels), 3)
    ch_score = np.round(metrics.calinski_harabasz_score(x, labels), 3)
    db_score = np.round(metrics.davies_bouldin_score(x, labels), 3)

    return np.array([s_score, db_score, ch_score])


def print_internal_clustering_scores(sim_nr, algorithm_number, score_values, gt=False):
    """
    Printing of internal evaluation scores.

    @param sim_nr: integer, Simulation number
    @param algorithm_number: integer, algorithm number
    @param score_values: 1D array, scores' values
    @param gt: Boolean, True - evaluation on ground truth values, False - evaluation of a clustering method
    """

    if gt:
        title = 'GT'
    else:
        title = cs.algorithms_[algorithm_number]

    print(
        "Simulation" + str(sim_nr) + " - " + title + " - " + 'Silhouette: {: .3f}'.format(
            score_values[0]))
    print(
        "Simulation" + str(sim_nr) + " - " + title + " - " + 'D-B: {: .3f}'.format(
            score_values[1]))
    print(
        "Simulation" + str(sim_nr) + " - " + title + " - " + 'C-H: {: .3f}'.format(
            score_values[2]))

