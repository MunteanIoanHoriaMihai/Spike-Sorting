import csv

from feature_extraction.weighted_pca.clustering import apply_clustering_algorithm
from feature_extraction.weighted_pca.feature_statistics import get_distribution_peaks, compute_average_divergence, \
    compute_divergence_matrix
from feature_extraction.weighted_pca.PCA_feature_extraction import apply_feature_extraction
from utils.benchmark.clustering_evaluation import print_external_clustering_scores, external_clustering_validation, \
    internal_clustering_validation, print_internal_clustering_scores
import utils.constants as cs
from utils.dataset_parsing.simulations_dataset import get_dataset_simulation
from utils.visualization import plot_clusters

from feature_extraction.weighted_pca.weighting import apply_weights
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr
import seaborn as sns
import os

prominence = 50
bin_method = 'fd'


def compute_min_max_list_arrays(X):
    mins = []
    maxs = []
    for sample in X:
        mins.append(min(sample))
        maxs.append(max(sample))

    print(min(mins))
    print(max(maxs))


def get_suffix(n):
    if n is 1:
        return "_wl"
    else:
        if n is 2:
            return "_wq"
        else:
            if n is 3:
                return "_p3"
            else:
                if n is 4:
                    return "_p4"


def get_alg_index(alg):
    if alg == 'K-Means':
        return 0
    elif alg == 'SBM':
        return 1
    elif alg == 'GMM':
        return 2
    elif alg == 'Ag':
        return 3


def print_correlation(X):
    for i in range(0, 79):
        for j in range(0, 79):
            corr, _ = pearsonr(X[:, i], X[:, j])
            corr = round(corr, 5)
            print(corr)


def statistic_analysis(feature):
    # range
    print(feature.min())
    print(feature.max())

    # variance
    print(feature.var())

    # density plot
    sns.displot(feature, kind="kde")


def do_spike_sorting(sim_nr, feature_extraction_method=None, nb_components=2, do_plot=True,
                     do_save=False,
                     folder="", nr_features=None, weighted_method=False, alg='K-Means',
                     scaling=False, minmax=False, divergence_file="", use_divergence=False, div_method=None):
    """
    Main Spike Sorting function.

    @param sim_nr: integer, simulation number
    @param feature_extraction_method: feature extraction method
    @param nb_components: integer, number of features after feature extraction
    @param do_plot: Boolean, to plot the clustering
    @param do_save: Boolean, to save the clustering plot
    @param folder: string, location to save the clustering plot
    @param nr_features: integer, number of features for feature selection considering the number of peaks
    @param weighted_method: Boolean, if the feature extraction method considers weighting the features
    @param alg: string, clustering algorithm name
    @param scaling: Boolean, if feature scaling is used
    @param minmax: Boolean, if MinMaxScaling is used
    @param divergence_file: string, location of divergence matrix
    @param use_divergence: Boolean, to use divergence if the weighting is used
    @param div_method: string, divergence method (kl, js or js_dist)
    @return: internal_performance_scores - list of internal performance scores (ARI, AMI or ARI, AMI, ARI-nnp, AMI-nnp)
             external_performance_scores - list of external performance scores (Silhouette, Davies-Boudlin, Calinski-Harabasz)
             X - 2D array, modified dataset after feature extraction
             y - 1D array, has the clustering label for each data sample
    """

    title_s = str(sim_nr)

    try:
        X, y = get_dataset_simulation(sim_nr, align_to_peak=True)

        if scaling:
            if minmax:
                std = MinMaxScaler()
            else:
                std = StandardScaler()
            X = std.fit_transform(X)

        # weighting
        if weighted_method is True:
            title_s += "_w"

            nr_features = min(nr_features, X.shape[1])
            ordered_data, peaks, ordered_indexes_list, ordered_peaks = get_distribution_peaks(X, nr_features,
                                                                                              bin_method,
                                                                                              prominence)
            div = []
            if use_divergence:
                divergence_tuples = compute_average_divergence(divergence_file, 79)
                d = dict(divergence_tuples)

                for i in range(len(divergence_tuples)):
                    div.append(d[i])
            else:
                div = None

            X, weights = apply_weights(X, peaks, div)

            if use_divergence:
                title_s += "_" + div_method

        # feature extraction
        if feature_extraction_method is not None:
            if alg == '':
                title_heat_map = "./demo/sim" + title_s
            else:
                title_heat_map = ""
            X = apply_feature_extraction(X, feature_extraction_method, nb_components=nb_components,
                                         title=title_heat_map, print_info=(alg == ''))

        labels = []

        if alg != '':
            a = get_alg_index(alg)

            # save in labels var the results of the clustering
            labels = apply_clustering_algorithm(X, y, a)

        # reduce dimension to visualize in 2D
        if X.shape[1] > 2:
            X_visualization = apply_feature_extraction(X, 'pca', nb_components=nb_components)
        else:
            X_visualization = X

        # visualization in 2D
        if X_visualization.shape[1] == 2:
            title_s = 'PCA 2D'
            filename = './' + folder + '/' + "sim" + title_s + "_gt" + '.png'
            plot_clusters(title_s, X_visualization, y, marker='o', plot=do_plot, save=do_save,
                 filename=filename)

            if alg != '':
                filename = './' + folder + '/' + "sim" + title_s + "_" + cs.algorithms_[a] + '.png'
                plot_clusters(cs.algorithms_[a] + " sim_" + title_s, X_visualization, labels, plot=do_plot, save=do_save,
                     filename=filename)

        # visualization in 3D
        elif X_visualization.shape[1] == 3:
            fig = px.scatter_3d(X_visualization, x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y)
            # fig.update_layout(title="Ground truth for Sim" + title_suffix)
            #
            # if do_plot:
            #     fig.show()
            #
            # fig = px.scatter_3d(X_visualization, x=X[:, 0], y=X[:, 1], z=X[:, 2], color=labels.astype(str))
            # fig.update_layout(title=cs.algorithms[a] + " for Sim" + title_suffix)
            #
            # if do_plot:
            #     fig.show()

        # performance evaluation
        if alg != '':
            external_performance_scores = external_clustering_validation(y, labels, no_noise_points=(a is 1))
            print_external_clustering_scores(sim_nr, a, external_performance_scores, no_noise_points=(a is 1))
        else:
            external_performance_scores = []

        if alg == '':
            internal_performance_scores = internal_clustering_validation(X, y)
            print_internal_clustering_scores(sim_nr, 0, internal_performance_scores, gt=True)
        else:
            internal_performance_scores = internal_clustering_validation(X, labels)
            print_internal_clustering_scores(sim_nr, a, internal_performance_scores, gt=False)

        return internal_performance_scores, external_performance_scores, X, y

    except:
        print('Could not read dataset')


def find_nr_clusters(sims):
    for s in sims:
        X, y = get_dataset_simulation(s, align_to_peak=True)
        print(y.max())


def compute_js_matrices():
    import time

    sims = [6, 7, 9, 10]

    for sim in sims:
        print(sim)
        start = time.time()
        X, _ = get_dataset_simulation(sim, align_to_peak=True)
        compute_divergence_matrix(X, method='js_div', base=2, bw_file='resources/bw_s' + str(sim) + '.txt',
                                  save_bw=True, save_matrix=True,
                                  filename='resources/js_div_matrix_s' + str(sim) + '_base2.txt')
        end = time.time()
        print('finished to compute js matrix in', (end - start))


def demo(sim):
    os.mkdir("demo/sim" + str(sim))
    algs = ['', 'K-Means', 'SBM']
    header_data = [['Method', 'S', 'D-B', 'C-H'], ['Method', 'ARI', 'AMI', 'S', 'D-B', 'C-H'],
                   ['Method', 'ARI-a', 'AMI-a', 'ARI-nnp', 'AMI-nnp', 'S', 'D-B', 'C-H']]
    methods = ['PCA2D', 'PCA2D & w', 'PCA2D and w & JS_div']

    for i, a in enumerate(algs):
        if a is '':
            filename = "./demo/sim" + str(sim) + "/sim + " + str(sim) + "_gt.csv"
        else:
            filename = "./demo/sim" + str(sim) + "/sim + " + str(sim) + "_" + a + ".csv"

        print('After PCA 2D')
        internal_1, external_1, _, _ = do_spike_sorting(sim, feature_extraction_method="pca", nb_components=2,
                                                        weighted_method=True,
                                                        nr_features=79, do_plot=(i==0), do_save=True,
                                                        folder='demo/sim' + str(sim), alg=a,
                                                        scaling=False,
                                                        minmax=True,
                                                        divergence_file='resources/' + 'js_div' + '_matrix_s' + str(
                                                            sim) + '_base2.txt',
                                                        use_divergence=False,
                                                        div_method='js_div'
                                                        )
        print('After weighting with peaks')
        internal_2, external_2, _, _ = do_spike_sorting(sim, feature_extraction_method="pca", nb_components=2,
                                                        weighted_method=True,
                                                        nr_features=79, do_plot=(i==0), do_save=True,
                                                        folder='demo/sim' + str(sim), alg=a,
                                                        scaling=False,
                                                        minmax=True,
                                                        divergence_file='resources/' + 'js_div' + '_matrix_s' + str(
                                                            sim) + '_base2.txt',
                                                        use_divergence=False,
                                                        div_method='js_div'
                                                        )

        print('After weighting with peaks and divergence')
        internal_3, external_3, _, _ = do_spike_sorting(sim, feature_extraction_method="pca", nb_components=2,
                                                        weighted_method=True,
                                                        nr_features=79, do_plot=(i==0), do_save=True,
                                                        folder='demo/sim' + str(sim), alg=a,
                                                        scaling=False,
                                                        minmax=True,
                                                        divergence_file='resources/' + 'js_div' + '_matrix_s' + str(
                                                            sim) + '_base2.txt',
                                                        use_divergence=True,
                                                        div_method='js_div'
                                                        )

        rows = [header_data[i]]
        if i == 0:
            rows.append([methods[0]] + [internal_1[0]] + [internal_1[1]] + [internal_1[2]])
            rows.append([methods[1]] + [internal_2[0]] + [internal_2[1]] + [internal_2[2]])
            rows.append([methods[2]] + [internal_3[0]] + [internal_3[1]] + [internal_3[2]])
        elif i == 1:
            rows.append(
                [methods[0]] + [external_1[0]] + [external_1[1]] + [internal_1[0]] + [internal_1[1]] + [internal_1[2]])
            rows.append(
                [methods[1]] + [external_2[0]] + [external_2[1]] + [internal_2[0]] + [internal_2[1]] + [internal_2[2]])
            rows.append(
                [methods[2]] + [external_3[0]] + [external_3[1]] + [internal_3[0]] + [internal_3[1]] + [internal_3[2]])
        elif i == 2:
            rows.append([methods[0]] +
                        [external_1[0]] + [external_1[1]] + [external_1[2]] + [external_1[3]] + [internal_1[0]] + [
                            internal_1[1]] + [internal_1[2]])
            rows.append([methods[1]] + [external_2[0]] + [external_2[1]] + [external_2[2]] + [external_2[3]] + [
                internal_2[0]] + [
                            internal_2[1]] + [internal_2[2]])
            rows.append([methods[2]] + [external_3[0]] + [external_3[1]] + [external_3[2]] + [external_3[3]] + [
                internal_3[0]] + [
                            internal_3[1]] + [internal_3[2]])

        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(rows)


if __name__ == '__main__':
    sim = 46
    internal_3, external_3, _, _ = do_spike_sorting(sim, feature_extraction_method="pca", nb_components=2,
                                                    weighted_method=False,
                                                    nr_features=79, do_plot=True, do_save=False,
                                                    folder='demo/sim' + str(sim), alg='',
                                                    scaling=False,
                                                    minmax=True,
                                                    divergence_file='resources/' + 'js_div' + '_matrix_s' + str(
                                                        sim) + '_base2.txt',
                                                    use_divergence=False,
                                                    div_method='js_div'
                                                    )
