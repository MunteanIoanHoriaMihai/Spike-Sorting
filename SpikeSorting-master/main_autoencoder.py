import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

from feature_extraction.autoencoder.softplusplus_activation import softplusplus
from utils.sbm import SBM, SBM_graph_merge, SBM_graph
from utils.dataset_parsing import simulations_dataset as ds
from utils import scatter_plot
from feature_extraction.autoencoder.model_auxiliaries import verify_output, get_codes, verify_random_outputs, \
    get_reconstructions
from feature_extraction.autoencoder.autoencoder import AutoencoderModel
from utils.scatter_plot import plot_spikes


def get_activation_function(activation_function):
    if activation_function == 'tanh':
        return activation_function
    elif activation_function == 'spp':
        return softplusplus


def get_loss_function(loss_function):
    if loss_function == 'mse':
        return loss_function
    elif loss_function == 'cce':
        return tf.keras.losses.CategoricalCrossentropy()
    elif loss_function == 'scce':
        return tf.keras.losses.SparseCategoricalCrossentropy()
    elif loss_function == 'bce':
        return tf.keras.losses.BinaryCrossentropy()


def spike_scaling_min_max(spikes, min_peak, max_peak):
    spikes_std = np.zeros_like(spikes)
    for col in range(len(spikes[0])):
        spikes_std[:, col] = (spikes[:, col] - min_peak) / (max_peak - min_peak)

    return spikes_std


def spike_scaling_ignore_amplitude(spikes):
    spikes_std = np.zeros_like(spikes)
    for row in range(len(spikes)):
        min_peak = np.amin(spikes[row])
        max_peak = np.amax(spikes[row])
        spikes_std[row] = (spikes[row] - min_peak) / (max_peak - min_peak)

    return spikes_std


def spike_scale_largest_amplitude(spikes):
    spikes_std = np.zeros_like(spikes)
    max_peak = np.amax(spikes)
    for col in range(len(spikes[0])):
        spikes_std[:, col] = spikes[:, col] / max_peak

    return spikes_std


def get_spike_energy(spike):
    return np.sum(np.power(spike, 2), axis=1)


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
simulation_number = 4
autoencoder_layer_sizes = [70, 60, 50, 40, 30, 20, 10, 5]
code_size = 2
PLOT_PATH = './feature_extraction/autoencoder/testfigs/'
MODEL_PATH = './feature_extraction/autoencoder/weights/'

spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)
# plot_spikes(spikes, title='scale_no', path=PLOT_PATH, show=False, save=True)
# spikes_scaled = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes))
# plot_spikes(spikes_scaled, title='scale_min_max', path=PLOT_PATH, show=False, save=True)
# spikes_scaled = (spikes_scaled * 2) - 1
# plot_spikes(spikes_scaled, title='scale_mod_-1_1', path=PLOT_PATH, show=False, save=True)
# spikes_scaled = spike_scaling_min_max(spikes, min_peak=-1, max_peak=1)
# plot_spikes(spikes_scaled, title='scale_-1_1', path=PLOT_PATH, show=False, save=True)
# spikes_scaled = preprocessing.MinMaxScaler((0, 1)).fit_transform(spikes)
# plot_spikes(spikes_scaled, title='scale_sklearn_0_1', path=PLOT_PATH, show=False, save=True)

# spikes_scaled = spike_scaling_ignore_amplitude(spikes)
# spikes_scaled = (spikes_scaled * 2) - 1
# plot_spikes(spikes_scaled, title='scale_no_amplitude', path=PLOT_PATH, show=False, save=True)
print(spikes.shape)



output_activation = 'tanh'
# output_activation = 'spp'

# loss_function = 'mse'
# loss_function = 'cce'
# loss_function = 'scce'
loss_function = 'bce'

nr_epochs = 100

spikes, labels = shuffle(spikes, labels, random_state=None)
spikes_scaled = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes))

# SCALE IGNORE AMPLITUDE ADDED FEATURE AMPLITUDE
# amplitudes = np.amax(spikes, axis=1)
# amplitudes = amplitudes.reshape((-1,1))
# spikes_scaled = spike_scaling_ignore_amplitude(spikes)
# spikes = (spikes_scaled * 2) - 1
# spikes = np.hstack((spikes, amplitudes))

# SCALE IGNORE AMPLITUDE
# spikes_scaled = spike_scaling_ignore_amplitude(spikes)

# SCALED ADDED FEATURE ENERGY
# spikes_energy = get_spike_energy(spikes)
# spikes_energy = spikes_energy.reshape((-1,1))
# spikes = np.hstack((spikes, spikes_energy))
# print(spikes.shape)

autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                               encoder_layer_sizes=autoencoder_layer_sizes,
                               decoder_layer_sizes=autoencoder_layer_sizes,
                               code_size=code_size,
                               output_activation=get_activation_function(output_activation),
                               loss_function=get_loss_function(loss_function))
autoencoder.train(spikes, epochs=nr_epochs)
autoencoder.save_weights(
    MODEL_PATH + f'autoencoder_cs{code_size}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')
# autoencoder.load_weights('./feature_extraction/autoencoder/weights/autoencoder_cs{code_size}_oa-tanh_sim4_e100')
encoder, decoder = autoencoder.return_encoder()

verify_output(spikes, encoder, decoder, path=PLOT_PATH)
autoencoder_features = get_codes(spikes, encoder)

print(autoencoder_features.shape)

scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
plt.savefig(PLOT_PATH + f'gt_model_cs{code_size}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')

pn = 25
sbm_labels = SBM.best(autoencoder_features, pn, ccThreshold=5, version=2)
scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, sbm_labels, marker='o')
plt.savefig(PLOT_PATH + f'gt_model_cs{code_size}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}_sbm')



# pca_2d = PCA(n_components=2)
# pca_features = pca_2d.fit_transform(spikes)
# scatter_plot.plot('GT' + str(len(pca_features)), pca_features, labels, marker='o')
# plt.savefig(PLOT_PATH + f'gt_pca_sim{simulation_number}')
#
# autoencoder_model = autoencoder.return_autoencoder()
# autoencoder_reconstructions = get_reconstructions(spikes, autoencoder_model)
#
# pca_2d = PCA(n_components=2)
# pca_reconstruction_features = pca_2d.fit_transform(autoencoder_reconstructions)
#
# scatter_plot.plot('GT' + str(len(pca_reconstruction_features)), pca_reconstruction_features, labels, marker='o')
# plt.savefig(PLOT_PATH + f'gt_pca_reconstruction_sim{simulation_number}')