import csv
import pickle
import struct

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyEMD import EMD
from scipy import fft
from scipy.fftpack import hilbert
from sklearn.decomposition import PCA

from utils.benchmark import benchmark_data as bd
from utils import constants as cs, scatter_plot
from utils.dataset_parsing import simulations_dataset as ds
import libraries.SimpSOM as sps
import libraries.som as som2
from feature_extraction.pipeline import pipeline
from feature_extraction import shape_features, feature_extraction_methods as fe
from utils.dataset_parsing.realdata_constants import spikes_per_channel, units_per_channel_5, units_per_channel

def get_mutual_info(simulation_nr):
    spikes, labels = ds.get_dataset_simulation(simulation_nr)
    # features = spike_features.get_features(spikes)
    features = shape_features.get_shape_phase_distribution_features(spikes)
    pca_2d = PCA(n_components=2)
    features = pca_2d.fit_transform(features)
    dims = ['fd_max', 'spike_max', 'fd_min']
    loading_scores = pd.Series(pca_2d.components_[0], index=dims)
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    print(sorted_loading_scores)

    # sbm_labels = SBM.parallel(features, pn=25, version=2)
    # res = dict(zip(["fd_max", "fd_min"],
    #                mutual_info_classif(features, sbm_labels, discrete_features="auto")
    #                ))
    # print(res)
    # print(mutual_info_classif(features, sbm_labels, discrete_features="auto"))


def generate_som(simulation_nr, dim, start_learn_rate, epochs):
    spikes, labels = ds.get_dataset_simulation(simulation_nr)

    filename = 'models/kohonen' + str(simulation_nr) + '_' + str(dim) + 'x' + str(dim) + '_' + str(epochs) + 'e.sav'

    try:
        k_map = pickle.load(open(filename, 'rb'))
    except FileNotFoundError:
        k_map = sps.somNet(dim, dim, spikes, PBC=True, PCI=True)
        k_map.train(startLearnRate=start_learn_rate, epochs=epochs)
        pickle.dump(k_map, open(filename, 'wb'))

    return spikes, labels, k_map


def som_features(simulation_nr, dim, start_learn_rate, epochs):
    spikes, labels, k_map = generate_som(simulation_nr, dim, start_learn_rate, epochs)
    features = np.array(k_map.project(spikes, show=True, printout=True))
    return spikes, labels, k_map, features


def som_metrics(simulation_nr, dim, start_learn_rate, epochs, show=True):
    spikes, labels, k_map, features = som_features(simulation_nr, dim, start_learn_rate, epochs)
    print("ala")
    alg_labels = [[], [], []]
    for alg in range(0, 3):
        alg_labels[alg] = bd.apply_algorithm(features, labels, alg)

    pe_labeled_data_results = [[], [], []]
    for alg in range(0, 3):
        pe_labeled_data_results[alg] = bd.benchmark_algorithm_labeled_data(labels, alg_labels[alg])
        bd.print_benchmark_labeled_data(simulation_nr, alg, pe_labeled_data_results[alg])
        bd.write_benchmark_labeled_data(simulation_nr, 'kohonen', pe_labeled_data_results)

    if show:
        filename_png = 'figures/kohonen' + str(simulation_nr) + '_' + str(dim) + 'x' + str(dim) + '_' + str(epochs) + \
                       'e.png'
        k_map.diff_graph(show=True, printout=True, filename=filename_png)

        scatter_plot.plot("Ground truth for Sim_" + str(simulation_nr), features, labels, marker='o')
        plt.show()
        for a in range(0, 3):
            scatter_plot.plot(cs.algorithms[a] + " on Sim_" + str(simulation_nr), features, alg_labels[a], marker='o')
            plt.show()


def som_err_graph(simulation_nr, dim, start_learn_rate, epochs):
    if __name__ == '__main__':
        spikes, labels = ds.get_dataset_simulation(simulation_nr)

        som = som2.SOM(dim, dim, alpha_start=start_learn_rate)  # initialize the SOM
        som.fit(spikes, epochs, save_e=True,
                interval=100)  # fit the SOM for 10000 epochs, save the error every 100 steps
        filename = 'figures/k_err' + str(simulation_nr) + '_' + str(dim) + 'x' + str(dim) + '_' + str(
            epochs) + 'e.png'
        som.plot_error_history(filename=filename)  # plot the training error history
        return som


# som_metrics(simulation_nr=22, dim=35, start_learn_rate=0.1, epochs=6500)
# som_err_graph(simulation_nr=2, dim=40, start_learn_rate=0.1, epochs=10000)

# print("Mutual Info alg", a, " ", mutual_info_classif(X, labels[a], discrete_features="auto"))
# get_mutual_info(21)


def get_features_shape_phase_distribution(spikes):
    pca_2d = PCA(n_components=2)

    features = shape_features.get_shape_phase_distribution_features(spikes)
    features = pca_2d.fit_transform(features)
    print("Variance Ratio = ", np.sum(pca_2d.explained_variance_ratio_))

    return features


def write_cluster_info(sim_nr_left, sim_nr_right):
    results = []
    for sim_nr in range(sim_nr_left, sim_nr_right + 1):
        if sim_nr == 25 or sim_nr == 44:
            continue
        print("Processing sim", sim_nr)
        spikes, labels = ds.get_dataset_simulation(sim_nr)
        for i in range(1 + max(labels)):
            cluster_spikes, cluster_labels = pipeline.generate_dataset_from_simulations2([sim_nr], [[i]])
            cluster_features = {"sim_nr": sim_nr, "spike_nr": i}
            cluster_features.update(shape_features.describe_cluster(cluster_spikes))
            results.append(cluster_features)
    with open('./results/Sim_%s_%s_features.csv' % (sim_nr_left, sim_nr_right), 'w', newline='') as file:
        writer = csv.DictWriter(file, results[0].keys())
        writer.writeheader()
        writer.writerows(results)


# def test_silhouette_sample(spikes, labels):
#     sil_coeffs = metrics.silhouette_samples(spikes, labels, metric='manhattan')
#     means = []
#     for label in range(max(labels) + 1):
#         means.append(sil_coeffs[labels == label].mean())
#     for i in np.arange(len(means)):
#         print(means[i])



# for i in range(1, 32):
#     try:
#         real_dataset(i, 'shape', 'pca2d')
#     except TypeError:
#         print('Error at ', i)
# write_cluster_info(1, 79)

def run_sim(sim_nr):
    bd.accuracy_all_algorithms_on_simulation(simulation_nr=sim_nr,
                                             feature_extract_method='hilbert',
                                             dim_reduction_method='derivatives_pca2d',
                                             # dim_reduction_method='pca2d',
                                             plot=True,
                                             pe_labeled_data=True,
                                             pe_unlabeled_data=False,
                                             pe_extra=False,
                                             # save_folder='kohonen',

                                             # som_dim=[20, 20],
                                             # som_epochs=1000,
                                             # title='sim' + str(sim_nr),
                                             # extra_plot=True,
                                             )


def run_pipeline():
    spikes, labels = ds.get_dataset_simulation(64)
    pipeline.pipeline(spikes, labels, [
        ['hilbert', 'derivatives_pca2d', 'mahalanobis', 0.65],
        ['stft_d', 'PCA2D', 'mahalanobis', 0.65],
        # ['superlets', 'PCA2D', 'euclidean', 0.65],
    ])


# run_sim(64)
run_sim(40)
# if __name__ == "__main__":
#     run_sim(10, 24)
# run_pipeline()

# bd.accuracy_all_algorithms_on_multiple_simulations(1, 3, feature_extract_method='hilbert',
#                                                    reduce_dimensionality_method='derivatives_pca2d')
