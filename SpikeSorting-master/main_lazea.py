import functools

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import pairwise_distances_chunked
from sklearn.metrics.cluster.unsupervised import check_number_of_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y

from utils.dataset_parsing import simulations_dataset as ds, realdata_spikes
from utils import scatter_plot
from feature_extraction import feature_extraction_methods as fe


def silhouette_samples2(X, labels, metric='euclidean', **kwds):
    X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    check_number_of_labels(len(le.classes_), n_samples)

    kwds['metric'] = metric
    reduce_func = functools.partial(silhouette_reduce2,
                                    labels=labels, label_freqs=label_freqs)
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func,
                                              **kwds))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)

    denom = (label_freqs - 1).take(labels, mode='clip')
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    # sil_samples = inter_clust_dists - intra_clust_dists
    sil_samples = inter_clust_dists
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)


def silhouette_reduce2(D_chunk, start, labels, label_freqs):
    """Accumulate silhouette statistics for vertical chunk of X

    Parameters
    ----------
    D_chunk : shape (n_chunk_samples, n_samples)
        precomputed distances for a chunk
    start : int
        first index in chunk
    labels : array, shape (n_samples,)
        corresponding cluster labels, encoded as {0, ..., n_clusters-1}
    label_freqs : array
        distribution of cluster labels in ``labels``
    """
    # accumulate distances from each sample to each cluster
    clust_dists = np.zeros((len(D_chunk), len(label_freqs)),
                           dtype=D_chunk.dtype)
    for i in range(len(D_chunk)):
        clust_dists[i] += np.bincount(labels, weights=D_chunk[i],
                                      minlength=len(label_freqs))

    # intra_index selects intra-cluster distances within clust_dists
    intra_index = (np.arange(len(D_chunk)), labels[start:start + len(D_chunk)])
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]
    # of the remaining distances we normalise and extract the minimum
    clust_dists[intra_index] = np.inf
    clust_dists /= label_freqs
    inter_clust_dists = clust_dists.min(axis=1)
    return intra_clust_dists, inter_clust_dists


def plot_spikes_from_cluster(simnr, spikes, labels, label):
    i = 0
    found = 0
    for spike in spikes:
        if labels[i] == label:
            found = found + 1
            plt.plot(np.arange(len(spikes[i])), spikes[i])
        i = i + 1
        if found == 20:
            break
    plt.title("Cluster_" + str(label) + "_Sim" + str(simnr))
    plt.show()


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def generate_dataset_from_simulations2(simulations, simulation_labels, save=False):
    spikes = []
    labels = []
    index = 0
    for sim_index in np.arange(len(simulations)):
        s, l = ds.get_dataset_simulation(simulations[sim_index], 79, True, False)
        for spike_index in np.arange(len(s)):
            for wanted_label in np.arange(len(simulation_labels[sim_index])):
                if simulation_labels[sim_index][wanted_label] == l[spike_index]:
                    spikes.append(s[spike_index])
                    labels.append(index + wanted_label)
        index = index + len(simulation_labels[sim_index])

    spikes = np.array(spikes)
    labels = np.array(labels)
    if save:
        np.savetxt("spikes.csv", spikes, delimiter=",")
        np.savetxt("labels.csv", labels, delimiter=",")

    return spikes, labels


def generate_dataset_from_simulations(simulations, simulation_labels, save=False):
    all_spikes = []
    all_labels = []
    for sim_index in np.arange(len(simulations)):
        s, l = ds.get_dataset_simulation(simulations[sim_index], 79, True, False)
        all_spikes.append(np.array(s))
        all_labels.append(np.array(l))
    spikes = []
    labels = []

    for sim_index in np.arange(len(all_spikes)):
        for spike_index in np.arange(len(all_spikes[sim_index])):
            # now all_spikes[sim_index][spike_index] will be the spike
            # check if the spike label is found in the wanted labels
            for wanted_label in np.arange(len(simulation_labels[sim_index])):
                if simulation_labels[sim_index][wanted_label] == all_labels[sim_index][spike_index]:
                    spikes.append(all_spikes[sim_index][spike_index])
                    labels.append(simulation_labels[sim_index][wanted_label])

    spikes = np.array(spikes)
    labels = np.array(labels)
    if save:
        np.savetxt("spikes.csv", spikes, delimiter=",")
        np.savetxt("labels.csv", labels, delimiter=",")

    return spikes, labels


def remove_separated_clusters_old(spikes, labels):
    labels_to_delete = []
    for i in np.arange(max(labels) + 1):
        label_to_keep = i
        labels_new = []
        for j in np.arange(len(labels)):
            if labels[j] == label_to_keep:
                labels_new.append(labels[j])
            else:
                labels_new.append(max(labels) + 2)
        silhouette_algorithm = metrics.silhouette_score(spikes, labels_new)
        # print("Label" + str(i) + " " + str(silhouette_algorithm))
        if silhouette_algorithm >= 0.3:
            labels_to_delete.append(label_to_keep)
            print("DELETED CLUSTER " + str(label_to_keep))
    new_spikes = []
    new_labels = []
    for i in np.arange(len(labels)):
        if labels[i] not in labels_to_delete:
            new_spikes.append(spikes[i])
            new_labels.append(labels[i])
    return np.array(new_spikes), np.array(new_labels)


def main():
    # sim_nr = 64
    # bd.accuracy_all_algorithms_on_simulation(simulation_nr=sim_nr,
    #                                          feature_extract_method='PCA2D',
    #                                          dim_reduction_method=None,
    #                                          plot=True,
    #                                          pe_labeled_data=True,
    #                                          pe_unlabeled_data=False,
    #                                          pe_extra=False
    #                                          # save_folder='EMD',
    #                                          # title='IMF_derivatives_PCA2D',
    #                                         )
    # fe_methods = ['derivatives2d', 'fsde', 'dwt']
    # for method in fe_methods:
    #     bd.accuracy_all_algorithms_on_simulation(simulation_nr=sim_nr,
    #                                              feature_extract_method=method,
    #                                              dim_reduction_method='pca2d',
    #                                              plot=True,
    #                                              pe_labeled_data=True,
    #                                              pe_unlabeled_data=False,
    #                                              pe_extra=False
    #                                              # save_folder='EMD',
    #                                              # title='IMF_derivatives_PCA2D',
    #                                              )

    # pipeline.pipeline(spikes, labels, [
    #         ['dwt', 'PCA3D', 'euclidean', 0.61],
    #         ['derivatives3d', None, 'euclidean', 0.60],
    #         ])

    # waveforms_ = read_waveforms('./datasets/real_data/Units/M017_0004_5stdv.ssduw')
    # waveforms_ = realdata.read_waveforms('C:/poli/year4/licenta_2021/codeGoodGit/Dissertation/datasets/realdata2/_Date spike sorting 19.06.2020/Sortate/M017_004_3stdv/Units/M017_S001_SRCS3L_25,50,100_0004.ssduw')
    # channel_nr = 16
    # spikes, labels = realdata.get_spike_units(waveforms_, channel=channel_nr)
    # spikes3d = fe.apply_feature_extraction_method(spikes, 'dwt', 'pca3d')
    # print(spikes3d.shape)
    # scatter_plot.plot_clusters(spikes3d, labels, 'dwt_Channel_' + str(channel_nr), "real_data")
    # plt.show()

    kampff_spike_len = 78
    waveforms_ = realdata_spikes.read_waveforms(
        'C:/poli/year4/licenta_2021/codeGoodGit/Dissertation/datasets/c37_npx/c37_npx.spikew')
    spikes = []
    for i in np.arange(len(waveforms_)/kampff_spike_len):
        spk = []
        for j in np.arange(kampff_spike_len):
            spk.append(-waveforms_[int(i * kampff_spike_len + j)])
        spikes.append(spk)
    # scatter_plot.plot_spikes(spikes, step=20, title="cell 37 - kampff dataset")
    features = fe.apply_feature_extraction_method(spikes,'cwt', 'pca2d')
    labels = np.ones(4811)
    scatter_plot.plot_clusters(features, labels, title="cell 37 kampff cwt + pca2d",
                               save_folder='')
main()
