import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from utils.benchmark import benchmark_data as bd
from utils import constants as cs, scatter_plot as sp
from utils.dataset_parsing import simulations_dataset as ds
from utils.constants import LABEL_COLOR_MAP

def gui():
    bd.accuracy_all_algorithms_on_simulation(simulation_nr=4,
                                             feature_extract_method='stft',
                                             dim_reduction_method='pca2d',
                                             plot=True,
                                             pe_labeled_data=True,
                                             pe_unlabeled_data=True,
                                             pe_extra=False,
                                             save_folder='demo',
                                             )


def plot_all_ground_truths():
    pca_2d = PCA(n_components=2)
    for sim_nr in range(95, 96):
        spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
        clusters = max(labels)
        signal_pca = pca_2d.fit_transform(spikes)
        sp.plot(title="GT with PCA Sim_%d (%d clusters)" % (sim_nr, clusters), X=signal_pca, labels=labels, marker='o')
        plt.savefig('./figures/sim_%d_c%d' % (sim_nr, clusters))
        # plt.show()
        # print(max(labels))


def spikes_per_cluster(sim_nr):
    sim_nr = sim_nr
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    print(spikes.shape)

    pca2d = PCA(n_components=2)

    for i in range(np.amax(labels) + 1):
        spikes_by_color = spikes[labels == i]
        print(len(spikes_by_color))
        sp.plot_spikes(spikes_by_color, "Sim_%d_Cluster_%d" % (sim_nr, i))
        cluster_pca = pca2d.fit_transform(spikes_by_color)
        # sp.plot(title="GT with PCA Sim_%d" % sim_nr, X=cluster_pca, marker='o')
        plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], c=LABEL_COLOR_MAP[i], marker='o', edgecolors='k')
        plt.title("Cluster %d Sim_%d" % (i, sim_nr))
        plt.savefig('figures/spikes_on_cluster/Sim_%d_Cluster_%d_color' % (sim_nr, i))
        plt.show()
        # print(cluster_pca)


def all_spikes():
    for sim_nr in range(77, 96):
        if sim_nr != 25 and sim_nr != 44:
            spikes_per_cluster(sim_nr)


def csf_db():
    pca_2d = PCA(n_components=2)
    alg_labels = [[], [], []]
    pe_labeled_data_results = [[], [], []]

    header_labeled_data = ['Simulation', 'Clusters', 'Algorithm', 'Index', 'Value']
    # with open('./results/PCA_2d_DBD.csv', 'w', newline='') as file:
    #     writer = csv.writer(file, delimiter=',')
    #     writer.writerows(header_labeled_data)

    for sim_nr in range(95, 96):
        spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
        signal_pca = pca_2d.fit_transform(spikes)

        for alg_nr in range(0, 3):
            alg_labels[alg_nr] = bd.apply_algorithm(signal_pca, labels, alg_nr)
            pe_labeled_data_results[alg_nr] = bd.benchmark_algorithm_labeled_data(labels, alg_labels[alg_nr])

        formatted_kmeans = ["%.3f" % number for number in pe_labeled_data_results[0]]
        formatted_dbscan = ["%.3f" % number for number in pe_labeled_data_results[1]]
        formatted_sbm = ["%.3f" % number for number in pe_labeled_data_results[2]]
        row1 = [sim_nr, max(labels), 'K-means', "ari_all", formatted_kmeans[0]]
        row2 = [sim_nr, max(labels), 'K-means', "ami_all", formatted_kmeans[1]]
        row3 = [sim_nr, max(labels), 'K-means', "ari_nnp", formatted_kmeans[2]]
        row4 = [sim_nr, max(labels), 'K-means', "ami_nnp", formatted_kmeans[3]]
        row5 = [sim_nr, max(labels), 'SBM', "ari_all", formatted_sbm[0]]
        row6 = [sim_nr, max(labels), 'SBM', "ami_all", formatted_sbm[1]]
        row7 = [sim_nr, max(labels), 'SBM', "ari_nnp", formatted_sbm[2]]
        row8 = [sim_nr, max(labels), 'SBM', "ami_nnp", formatted_sbm[3]]
        row9 = [sim_nr, max(labels), 'DBSCAN', "ari_all", formatted_dbscan[0]]
        row10 = [sim_nr, max(labels), 'DBSCAN', "ami_all", formatted_dbscan[1]]
        row11 = [sim_nr, max(labels), 'DBSCAN', "ari_nnp", formatted_dbscan[2]]
        row12 = [sim_nr, max(labels), 'DBSCAN', "ami_nnp", formatted_dbscan[3]]
        row_list = [row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12]
        with open('./results/PCA_2d_DBD.csv', 'a+', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(row_list)


def save_all_pca2d():
    pca_2d = PCA(n_components=2)

    for alg_nr in range(2, 3):
        average = [0, 0, 0, 0, 0]
        # average = [0, 0, 0, 0, 0, 0]
        simulation_counter = 0
        for sim_nr in range(1, 96):
            if sim_nr != 25 and sim_nr != 27 and sim_nr != 44:
                spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2,
                                                           normalize_spike=False)
                signal_pca = pca_2d.fit_transform(spikes)
                # signal_pca = derivatives.compute_fdmethod(spikes)
                alg_labels = bd.apply_algorithm(signal_pca, labels, alg_nr)
                results = bd.benchmark_algorithm_labeled_data(labels, alg_labels)
                # results = bd.benchmark_algorithm_extra(alg_labels, labels)
                simulation_counter += 1
                average += results
                formatted = ["%.3f" % number for number in results]
                row = [sim_nr, formatted[0], formatted[1], formatted[2], formatted[3], formatted[4]]
                # row = [sim_nr, formatted[0], formatted[1], formatted[2], formatted[3], formatted[4], formatted[5]]
                with open('./results/all_%s_pca3d.csv' % cs.algorithms[alg_nr], 'a+', newline='') as file:
                    writer = csv.writer(file, delimiter=',')
                    writer.writerow(row)
        average = average / simulation_counter
        with open('./results/all_%s_pca3d.csv' % cs.algorithms[alg_nr], 'a+', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(average)


def sim_details():
    for sim_nr in range(45, 96):
        spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
        row = [sim_nr, max(labels), spikes.shape[0]]
        with open('./results/simulaton_details.csv', 'a+', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(row)


def plot_single_spike():
    sim_nr = 15
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    spike = spikes[0]
    print(spike)
    plt.plot(np.arange(79), spike)
    plt.show()
    for i in range(1, 79):
        print('f(%d,%d)=1' % (i, spike[i]))



gui()
# plot_all_ground_truths()
# spikes_per_cluster(2)
# all_spikes()
# csf_db()
# sim_details()
# plot_single_spike()
# save_all_pca2d()
