import numpy as np




def remove_predicted_noise(true_labels, predicted_labels, noise_label):
    return true_labels[predicted_labels!=noise_label], predicted_labels[predicted_labels != noise_label]




def ss_metric(true_labels, predicted_labels, remove_noise=False):
    if isinstance(remove_noise, bool) and remove_noise == False:
        pass
    else:
        true_labels, predicted_labels = remove_predicted_noise(true_labels, predicted_labels, remove_noise)

    score = 0
    for unique_true_label in np.unique(true_labels):
        #print()
        only_true_label = true_labels == unique_true_label
        #print(unique_true_label, np.count_nonzero(only_true_label))
        predicted_labels_of_true = predicted_labels[only_true_label]
        #print(np.unique(predicted_labels_of_true, return_counts=True))

        predicted_unique_labels = np.unique(predicted_labels_of_true, return_counts=True)[0]
        predicted_counts = np.unique(predicted_labels_of_true, return_counts=True)[1]

        predicted_label_of_true = predicted_unique_labels[np.argmax(predicted_counts)]
        #print(predicted_label_of_true)

        score += np.amax(predicted_counts) / np.count_nonzero(predicted_labels == predicted_label_of_true)
        #print(f"Label {unique_true_label} - Score: {np.amax(predicted_counts) / np.count_nonzero(predicted_labels == predicted_label_of_true)}")
    return score / len(np.unique(true_labels))