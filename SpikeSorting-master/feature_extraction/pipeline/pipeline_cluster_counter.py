from collections import Counter

import numpy as np
from sklearn import metrics

from utils.dataset_parsing import simulations_dataset as ds
from feature_extraction import feature_extraction_methods as fe
from utils import scatter_plot


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


def remove_separated_clusters(features, spikes, labels, metric, threshold, cluster_counter, f):
    labels_to_delete = []

    print("Applying metric " + metric + " with threshold " + str(threshold))
    f.write("\nApplying metric " + metric + " with threshold " + str(threshold))

    sil_coeffs = metrics.silhouette_samples(features, labels, metric=metric)
    means = []
    for label in np.arange(max(labels) + 1):
        if label not in labels:
            means.append(-1)
        else:
            means.append(sil_coeffs[labels == label].mean())
    for j in np.arange(len(means)):
        if means[j] != -1:
            print(means[j])
            f.write("\n" + str(means[j]))
            if means[j] >= threshold:
                labels_to_delete.append(j)
                print("Deleted cluster " + str(j))
                f.write("\nDeleted cluster " + str(j))
                cluster_counter += 1
    new_spikes = []
    new_labels = []
    new_features = []
    for i in np.arange(len(labels)):
        if labels[i] not in labels_to_delete:
            new_spikes.append(spikes[i])
            new_labels.append(labels[i])
            new_features.append(features[i])

    return np.array(new_spikes), np.array(new_features), np.array(new_labels), cluster_counter


def pipeline(spikes, labels, methods, f, sim_nr):
    stop = False
    step = 0
    cluster_counter = 0
    while len(labels) > 0 and not stop:
        changed = False
        for method in methods:
            print("\nPipeline step " + str(step) + " applying " + method[0])
            f.write("\nPipeline step " + str(step) + " applying " + method[0])

            features = fe.apply_feature_extraction_method(spikes, method[0], method[1])
            new_spikes, new_features, new_labels, new_cluster_counter = remove_separated_clusters(features, spikes,
                                                                                                  labels, method[2],
                                                                                                  method[3],
                                                                                                  cluster_counter, f)
            cluster_counter = new_cluster_counter
            print(cluster_counter)
            f.write("\nEliminated: " + str(cluster_counter))
            scatter_plot.plot_clusters(features, labels, title="sim%d_step%d_using_%s" % (sim_nr, step, method[0]),
                                       save_folder='pipeline')

            if len(labels) != len(new_labels):
                changed = True
            labels = new_labels
            spikes = new_spikes
            step = step + 1
            if len(Counter(labels).keys()) <= 1:
                changed = False
                break
        stop = not changed


def cluster_count():
    clusters_2 = [8, 39, 46, 59, 94]
    clusters_3 = [14, 29, 53, 76, 89]
    clusters_4 = [4, 21, 33, 64, 83]
    clusters_5 = [30, 54, 77, 87]
    clusters_6 = [22, 24, 57, 62, 84]
    f = open("./results/pipeline/pipeline_stft_slt.txt", 'a')
    for sim_nr in clusters_6:
        f.write("\n\nSimulation " + str(sim_nr) + "\n")
        spikes, labels = generate_dataset_from_simulations2([sim_nr], [[0, 1, 2, 3, 4, 5, 6]], False)

        pipeline(spikes, labels, [
            # ['hilbert', 'derivatives2d', 'mahalanobis', 0.67],
            ['stft_d', 'PCA2D', 'mahalanobis', 0.67],
            ['superlets', 'PCA2D', 'euclidean', 0.70],
            # ['pca', 'PCA2D', 'euclidean', 0.70],
        ], f, sim_nr)
    f.close()


# spikes, labels = generate_dataset_from_simulations2([4], [[0, 1, 2, 3, 4]], False)
# spikes, labels = generate_dataset_from_simulations2([21], [[0, 1, 2, 3]], False)
# spikes, labels = generate_dataset_from_simulations2([1, 2, 6, 12, 24, 28, 2, 15, 17],
#                                                     [[10], [7], [6], [15], [2], [8], [13], [8], [2]], False)

cluster_count()

# pipeline(spikes, labels, [['pca2d', 'PCA2D', 'euclidean', 0.65]])
