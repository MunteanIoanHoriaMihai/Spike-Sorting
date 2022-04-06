from collections import Counter

import numpy as np
from sklearn import metrics

from feature_extraction import feature_extraction_methods as fe
from utils import scatter_plot


def remove_separated_clusters(features, spikes, labels, metric, threshold):
    labels_to_delete = []

    print("Applying metric " + metric + " with threshold " + str(threshold))

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
            if means[j] >= threshold:
                labels_to_delete.append(j)
                print("Deleted cluster " + str(j))
    new_spikes = []
    new_labels = []
    new_features = []
    for i in np.arange(len(labels)):
        if labels[i] not in labels_to_delete:
            new_spikes.append(spikes[i])
            new_labels.append(labels[i])
            new_features.append(features[i])

    return np.array(new_spikes), np.array(new_features), np.array(new_labels)


def pipeline(spikes, labels, methods):
    stop = False
    step = 0
    while len(labels) > 0 and not stop:
        changed = False
        for method in methods:
            print("\nPipeline step " + str(step) + " applying " + method[0])

            features = fe.apply_feature_extraction_method(spikes, method[0], method[1])
            new_spikes, new_features, new_labels = remove_separated_clusters(features, spikes, labels, method[2],
                                                                             method[3])

            scatter_plot.plot_clusters(features, labels, title="GT step %d using %s" % (step, method[0]),
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
