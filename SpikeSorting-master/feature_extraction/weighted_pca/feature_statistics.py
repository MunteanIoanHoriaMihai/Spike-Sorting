from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from preprocessing_dataset import get_dataset_simulation
from scipy.signal import find_peaks, peak_prominences
import numpy as np
from numpy import math
from matplotlib import pyplot as plt
import utils.constants as c
from ss.sshist import sshist
import scipy.stats
from warnings import simplefilter
from utils.helper_functions import write_2d_array_in_file, read_2d_array_from_file, print_2d_array
from utils.visualization import plot_distribution_one_feature

simplefilter(action='ignore', category=DeprecationWarning)
simplefilter(action='ignore', category=FutureWarning)


def compute_histogram_one_feature(X, feature_index, bins_method='auto'):
    """
    Creates the histogram of data stored in X for a specific feature. ☺☻♥

    @param X: 2D array of shape (n_samples, n_features), stores the data
    @param feature_index: integer, index of the feature for which the histogram is created
    @param bins_method:  string, bin method
    @return: 1D array, represents the density array obtained from the histogram
    """

    # select the data for a specific feature
    spikes = X[:, feature_index]

    # compute the histogram, 2 arrays: 0 - density (Oy); 1 - values (Ox)
    if bins_method is 'ss':
        bins, _, _, _, _ = sshist(spikes)
        # print(bins)
        distribution, _ = np.histogram(spikes, bins)
    else:
        distribution, bins_arr = np.histogram(spikes, bins_method)
        bins = len(bins_arr) - 1

    # save the density values
    values = distribution

    return values, bins


def compute_maxima(values, prominence=50):
    """
    Computes the number of peaks of a distribution.

    @param values: 1D array, stores the values of the distribution
    @param prominence: integer, prominence value
    @return: integer, the number of peaks of the distribution
    """

    # compute_prominences(values)
    peaks, _ = find_peaks(x=values, prominence=prominence)

    return len(peaks)


def compute_histogram(X):
    x = X[:, 30]
    nr_bin = 40

    bins = np.linspace(math.floor(x.min()), math.floor(x.max()), nr_bin)
    bins1 = np.linspace(math.floor(x.min()), math.floor(x.max()), 20)

    plt.hist(x, bins, alpha=0.9, label='x')
    # pyplot.hist(x, bins1, alpha=0.3, label='y')
    plt.legend(loc='upper right')
    plt.show()


def compute_prominences(x):
    peaks, _ = find_peaks(x=x)  # indices of peaks

    prominences = peak_prominences(x, peaks)[0]
    # prominences.sort()

    contour_heights = x[peaks] - prominences
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.vlines(x=peaks, ymin=contour_heights, ymax=x[peaks])
    plt.show()

    plt.plot(x)
    # for a, b in enumerate(x):
    #     plt.text(a, b, str(b))
    # plt.show()

    print(prominences)


def get_distribution_peaks(X, number_of_features, bin_method='auto', prominence=20):
    """
    Computes the number of peaks of the histograms for each feature of dataset X.

    @param X: 2D array of shape (n_samples, n_features), stores the data
    @param number_of_features: integer, subset of features ( < X.shape[1])
    @param bin_method: string, bin method name
    @param prominence: integer
    @return: extracted_features: matrix - reduced dataset, samples having only the most relevant features according to the number of peaks
             nr_peaks: array - the number of peaks for the most relevant subset of features
             indexes_list: sorted indexes descending according to peaks
             ordered_peaks:
    """

    # array of tuples (feature_index, distribution, nr_peaks, nr_bins)
    features_distributions = []

    # traverse each feature (X.shape[1] = 79)
    for i in range(X.shape[1]):
        values, bins = compute_histogram_one_feature(X, i,
                                                     bin_method)  # histogram for feature i; an array of bins dimension

        # values = histogram of values
        tops = compute_maxima(values, prominence)  # the number of peaks for that distribution/histogram

        features_distributions.append((i, values, tops, bins))

    # save the number of peaks for each feature, in the initial order!
    nr_peaks = [i[2] for i in features_distributions]

    # sort features in decreasing order of their peaks
    features_distributions.sort(key=lambda tup: tup[2], reverse=True)

    # select only a subset of features
    top_features = features_distributions[:number_of_features]

    ordered_peaks = [i[2] for i in top_features]

    # save the index of the features
    indexes_list = [i[0] for i in top_features]

    # column features ordered by peaks, descending
    extracted_features = X[:, indexes_list]

    return extracted_features, nr_peaks, indexes_list, ordered_peaks


def determine_best_bin_method(sim_nr, feature_nr):
    """
    Plots the histogram of a feature from the dataset of a specific simulation for each bin method from bin_method
    array stored in constants file.

    @param sim_nr: integer, simulation number
    @param feature_nr: integer, feature index
    """

    X, _ = get_dataset_simulation(sim_nr)

    bins, _, _, _, _ = sshist(X[:, feature_nr])
    print('sshist', bins)
    plot_distribution_one_feature(X, feature_nr, bins)

    for i in c.bin_method:
        vals, bins = compute_histogram_one_feature(X, feature_nr, i)
        print(i, bins)
        plot_distribution_one_feature(X, feature_nr, bins)


def find_kde_bw(x):
    """
    Find the best bandwidth for kernel density estimator from a range of values for x.

    @param x: 1D array, stores the data
    @return: float, bandwidth value
    """

    bandwidths = np.arange(0, 0.5, 0.005)

    grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, n_jobs=-1)

    grid.fit(x[:, None])

    return grid.best_params_['bandwidth']


def compute_kde(x, bw=0.05, show_plot=False, save_plot=False, save_dir="", filename=""):
    """

    @param x:
    @param bw:
    @param show_plot:
    @param save_plot:
    @param save_dir:
    @param filename:
    @return:
    """

    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(
        x[:, np.newaxis])

    if show_plot or save_plot:
        x_plot = np.linspace(x.min(), x.max(), x.shape[0])[:, np.newaxis]
        logprob = kde.score_samples(x_plot)
        label = 'bw=' + str(bw)
        density_curve, = plt.plot(x_plot, np.exp(logprob), label=label)

        plt.xlabel('magnitude', fontsize=15)
        plt.ylabel('Probability Density', fontsize=15)
        plt.legend(handles=[density_curve])

    if show_plot:
        plt.show()

    if save_plot:
        plt.savefig(save_dir + '/' + filename + '.png')
        plt.close()

    return kde


def compute_probability(start, end, eval_points, kd):
    """

    @param start:
    @param end:
    @param eval_points:
    @param kd:
    @return:
    """

    n = eval_points
    step = (end - start) / (n - 1)

    x = np.linspace(start, end, n)[:, np.newaxis]
    kd_vals = np.exp(kd.score_samples(x))  # get pdf values for each x

    # approximate the integral of pdf
    temp = kd_vals * step
    # temp = np.around(kd_vals * step, 3)

    # probs = np.sum(temp)
    # print(probs)

    return temp


def kl_divergence(p, q, log_base=2):
    """
    Computes Kullback-Leibler divergence of two probability distributions.

    @param p: first probability distribution
    @param q: second probability distribution
    @param log_base: logarithm base
    @return: KL divergence score
    """

    return sum(p[i] * (np.log2(p[i] / q[i]) / np.log2(log_base)) for i in range(len(p)))


def jensen_shannon_divergence(p, q, base=np.e):
    m = 0.5 * (p + q)
    # return 0.5 * kl_divergence(p, m, log_base) + 0.5 * kl_divergence(q, m, log_base)
    return 0.5 * scipy.stats.entropy(p, m, base) + 0.5 * scipy.stats.entropy(q, m, base)


def jensen_shannon_distance(p, q, base=np.e):
    return np.sqrt(jensen_shannon_divergence(p, q, base))


def divergence_test():
    p = np.array([0.10, 0.40, 0.50])
    q = np.array([0.80, 0.15, 0.05])

    print(np.e)

    # KL(P || Q)
    kl_pq = kl_divergence(p, q, np.e)
    print('KL(P || Q): %.3f bits' % kl_pq)

    # KL(Q || P)
    kl_qp = kl_divergence(q, p, np.e)
    print('KL(Q || P): %.3f bits' % kl_qp)

    # JS(P || Q) = JS(Q || P)
    js_div = jensen_shannon_divergence(p, q, 2)
    print('JS(P || Q): %.3f bits' % js_div)

    js_dist = jensen_shannon_distance(q, p, 2)
    print('JS(Q || P): %.3f bits' % js_dist)


def compute_range_all_features(X, save=False, filename=""):
    """
    Computes the range (min, max) of all features of dataset X.

    @param X: 2D array of shape (n_samples, n_features), stores the data
    @param save: Boolean, to save the ranges or not
    @param filename: string, location for saving
    @return:
    """

    ranges = np.zeros((X.shape[1], 2))

    for i in range(X.shape[1]):
        minimum = min(X[:, i])
        maximum = max(X[:, i])

        ranges[i][0] = minimum
        ranges[i][1] = maximum

    print_2d_array(ranges)

    if save:
        write_2d_array_in_file(ranges, filename)

    return ranges

    # print min max overall
    # print('min overall %f' % min(ranges[:, 0]))
    # print('max overall %f' % max(ranges[:, 1]))

    # plot distribution
    # sns.displot(ranges[:, 0], kde=True)
    # sns.displot(ranges[:, 1], kde=True)
    # plt.show()


def test_pdf():
    """
    Method for testing that pdf was computed well. The sum of probabilities has to be 1.
    """

    y = np.random.normal(10, 4, 100)
    kde = compute_kde(y)
    probs = compute_probability(min(y), max(y), 50, kde)
    print(np.sum(probs))  # should be very close to 1 (total area under the curve is 1)


def compute_probability_distribution_all_features(X, save_bw=False, filename=""):
    """
    Compute the probability distributions for each feature of X.

    @param X: 2D array of shape (n_samples, n_features), stores the data
    @param save_bw: Boolean, to save the bandwidth values
    @param filename: string, location to save the probabilities
    @return: list of probability distributions for each feature of dataset X
    """

    nb_events = X.shape[0]
    nb_features = X.shape[1]
    probabilities = []

    import os
    if os.path.isfile(filename) is True:
        save_bw = False

    f = None
    if save_bw is True:
        f = open(filename, "w")
    else:
        f = open(filename, "r")

    for i in range(nb_features):
        feature = X[:, i]

        if save_bw is True:
            bw = find_kde_bw(feature)
            # print(bw)
            f.write(str(bw))
            f.write('\n')
        else:
            temp = f.readline()
            bw = float(temp)
            # print(bw)

        kde = compute_kde(feature, bw, show_plot=False, save_plot=False, save_dir="",
                          filename='f' + str(i))
        # print(i)
        probability = compute_probability(start=min(feature), end=max(feature), eval_points=nb_events, kd=kde)
        probabilities.append(probability)

    f.close()

    return probabilities


def compute_divergence_matrix(X, method='kl', base=2, bw_file="", save_bw=False, save_matrix=False, filename=""):
    """
    Compute 79x79 divergence/distance matrix.

    @param X: 2D array of shape (n_samples, n_features), stores the data
    @param method: string, method for computing the divergence, kl, js, js_dist
    @param base: logarithm base
    @param bw_file: string, location to save the array of bandwidths
    @param save_bw: Boolean, to save the bandwidth of each feature
    @param save_matrix: Boolean, to save the divergence matrix
    @param filename: string, location to save the divergence matrix
    """

    matrix = np.zeros((X.shape[1], X.shape[1]))
    probabilities = compute_probability_distribution_all_features(X, save_bw=save_bw, filename=bw_file)

    nb_features = X.shape[1]

    for i in range(nb_features):
        for j in range(nb_features):
            if method == 'kl_div':
                matrix[i][j] = scipy.stats.entropy(probabilities[i], probabilities[j], base=base)
            elif method == 'js_div':
                matrix[i][j] = jensen_shannon_divergence(probabilities[i], probabilities[j], base=base)
            elif method == 'js_dist':
                matrix[i][j] = jensen_shannon_distance(probabilities[i], probabilities[j], base=base)

    if save_matrix:
        write_2d_array_in_file(matrix, filename)


def test_kde_bw(x):
    bandwidths = [0.05, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2]
    density_curves = []
    fig, ax = plt.subplots()
    # ax.hist(x, fc='gray', alpha=0.3)

    for i in bandwidths:
        kde = KernelDensity(kernel='gaussian', bandwidth=i).fit(x[:, np.newaxis])
        label = 'bw=' + str(i)
        x_plot = np.linspace(x.min(), x.max(), x.shape[0])[:, np.newaxis]

        density_curve, = ax.plot(x_plot, np.exp(kde.score_samples(x_plot)), label=label)
        density_curves.append(density_curve)

    plt.xlabel('magnitude', fontsize=15)
    plt.ylabel('probability density', fontsize=15)
    plt.legend(handles=density_curves)
    plt.savefig('s79f30.png')


def compute_average_divergence(filename="", n=0, sort=False):
    """
    Reads the divergence matrix from the specified location and compute the average divergence for each feature by
    computing the average on rows.

    @param filename: string, location where the divergence matrix is stored
    @param n: number of features
    @param sort: Boolean, to sort the array of divergences descending
    @return: 1D array of divergences for each feature
    """

    matrix = read_2d_array_from_file(filename, n, n)

    divergence = []

    # sum on rows
    for i in range(matrix.shape[0]):
        sum = 0
        for j in range(matrix.shape[1]):
            sum += matrix[i][j]

        divergence.append((i, sum))

    divergence = [(i, x / matrix.shape[0]) for (i, x) in divergence]

    if sort:
        divergence.sort(key=lambda tup: tup[1], reverse=True)

    return divergence


def compute_variance(X, filename=""):
    vars = []
    for f in range(0, 79):
        vars.append((f, np.var(X[:, f])))

    vars.sort(key=lambda tup: tup[1], reverse=True)

    vars_s = []
    indexes = []

    for tup in vars:
        vars_s.append(tup[1])
        indexes.append(tup[0])

    for i, j in enumerate(vars_s):
        j = round(j, 3)
        plt.annotate(str(j), xy=(i, j))
        if i == 2:
            break

    plt.plot(vars_s)
    plt.savefig(filename)
    plt.show()
