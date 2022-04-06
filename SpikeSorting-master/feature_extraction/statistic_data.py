import csv
import sys
import warnings

import numpy as np

from feature_extraction.weighted_pca.weighting import apply_weights
from utils.benchmark.benchmark_data import benchmark_algorithm_labeled_data, distribution_filter_features, apply_algorithm
from utils.dataset_parsing import simulations_dataset as ds
import feature_extraction.feature_extraction_methods as fe
import statistics as sts

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.setrecursionlimit(100000)

batch_size = 10




def statistic_kmeans_on_sim(simulation_nr, feature_extract_method, dim_reduction_method=None,
                            pe_labeled_data=True, save_folder="",
                            nr_features=None, weighted=False, **kwargs):
    print(f"Starting statistics on sim {simulation_nr}")

    # get original data
    X, y = ds.get_dataset_simulation(simulation_nr)

    title_suffix = str(simulation_nr)

    # reduce the feature space
    if feature_extract_method is not None:
        X = fe.apply_feature_extraction_method(X, feature_extract_method, dim_reduction_method, **kwargs)
        title_suffix = title_suffix + "_" + feature_extract_method

    if nr_features is not None:
        nr_features = min(nr_features, X.shape[1])
        print(f"Number of features for distribution is {nr_features}")
        X, peaks = distribution_filter_features(X, nr_features)
        if weighted is True:
            X = apply_weights(X, peaks)
            title_suffix = title_suffix + 'features_weighted'

    # apply algorithm(s) and save clustering labels
    labels = []
    scores = [[], [], [], []]
    for i in range(batch_size):
        labels = apply_algorithm(X, y, 0)
        pe_labeled_data_results = benchmark_algorithm_labeled_data(y, labels, False)
        scores[0].append(pe_labeled_data_results[0])
        scores[1].append(pe_labeled_data_results[1])
        scores[2].append(pe_labeled_data_results[2])

    mean_final = [sts.mean(scores[0]), sts.mean(scores[1]), sts.mean(scores[2])]
    std_dev_final = [sts.stdev(scores[0]), sts.stdev(scores[1]), sts.stdev(scores[2])]

    print(mean_final)
    print(std_dev_final)
    return mean_final, std_dev_final, len(np.unique(y))


def statistic_kmeans_on_sim_interval(simulation_nr_l, simulation_nr_r, feature_extract_method, dim_reduction_method=None,
                                     pe_labeled_data=True, save_folder="",
                                     nr_features=None, weighted=False, **kwargs):
    header_labeled_data = ['Sim', 'ARI-a', 'AMI-a', 'FMI-a', 'ARI-std',  'AMI-std', 'FMI-std', 'clusterNr']
    rows = [header_labeled_data]
    for i in range(simulation_nr_l, simulation_nr_r):
        mean, std_dev, clusters = statistic_kmeans_on_sim(i, feature_extract_method, dim_reduction_method,
                                                pe_labeled_data, save_folder,
                                                nr_features, weighted, **kwargs)
        print([[i] + mean + std_dev])
        rows.append([i] + mean + std_dev + [clusters])

    print(rows)
    with open(f'./results/Sim_{simulation_nr_l}_{simulation_nr_r}_labeled_{feature_extract_method}_{dim_reduction_method}_{weighted}.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(rows)

