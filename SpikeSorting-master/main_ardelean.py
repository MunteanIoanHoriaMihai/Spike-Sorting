import os
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import windows
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score, silhouette_score, homogeneity_completeness_v_measure

from feature_extraction.autoencoder import lstm_input, fft_input
from feature_extraction.autoencoder.benchmark import validate_model
from feature_extraction.autoencoder.fft_input import fft_padded_spike
from feature_extraction.autoencoder.lstm_autoencoder import LSTMAutoencoderModel
from feature_extraction.autoencoder.lstm_input import lstm_get_codes
from feature_extraction.autoencoder.softplusplus_activation import softplusplus
from utils.dataset_parsing.simulations_dataset_autoencoder import stack_simulations_array
from utils.dataset_parsing.simulations_dataset import get_dataset_simulation
from utils.sbm import SBM, SBM_graph_merge, SBM_graph
from utils.dataset_parsing import simulations_dataset as ds
from utils.dataset_parsing import simulations_dataset_autoencoder as dsa
from utils import scatter_plot
from utils.constants import autoencoder_layer_sizes, autoencoder_code_size, lstm_layer_sizes, lstm_code_size, autoencoder_cascade_layer_sizes, LABEL_COLOR_MAP, autoencoder_expanded_layer_sizes, \
    autoencoder_expanded_code_size, autoencoder_selected_layer_sizes, autoencoder_single_sim_layer_sizes, autoencoder_single_sim_code_size, lstm_single_sim_code_size, lstm_single_sim_layer_sizes
from feature_extraction.autoencoder.model_auxiliaries import verify_output, get_codes, verify_random_outputs
from feature_extraction.autoencoder.autoencoder import AutoencoderModel
from sklearn.tree import DecisionTreeClassifier

import matplotlib as mpl
mpl.rc('figure', max_open_warning = 0)

from utils.sbm.SBM_functions import distance



def main(program, sub=""):
    # data = ds.getTINSData()
    # data, y = ds.getGenData()
    if program == "noise_isolation":
        for simulation_number in range(1, 96):
            if simulation_number == 25 or simulation_number == 44:
                continue
            spike_features, labels = ds.get_dataset_simulation_features(simulation_number)
            scatter_plot.plot('GT' + str(len(spike_features)), spike_features, labels, marker='o')
            plt.savefig(f'./figures/noise/gt_model_sim{simulation_number}')
    elif program == "sbm":
        # data = ds.getTINSData()
        # data, y = ds.getGenData()
        data, y = ds.get_dataset_simulation_pca_2d(simNr=79, align_to_peak=True)

        pn = 25
        start = time.time()
        labels = SBM.best(data, pn, ccThreshold=5, version=2)
        end = time.time()
        print('SBM: ' + str(end - start))

        scatter_plot.plot_grid('SBM' + str(len(data)), data, pn, labels, marker='o')
        plt.savefig('./figures/SBMv2_sim79')


        scatter_plot.plot('GT' + str(len(data)), data, y, marker='o')
        plt.savefig('./figures/ground_truth')

        unique, counts = np.unique(labels, return_counts=True)
        print(dict(zip(unique, counts)))

        # plt.show()
    elif program == "sbm_graph":
        # data = ds.getTINSData()
        # data, y = ds.getGenData()
        spikes, y = get_dataset_simulation(79, 79, True, False)
        # apply pca_nonoise
        pca_5d = PCA(n_components=5)
        spik = pca_5d.fit_transform(spikes)
        pca_2d = PCA(n_components=2)
        data = pca_2d.fit_transform(spikes)

        pn = 25
        start = time.time()
        labels, pv = SBM_graph.SBM(spik, pn, ccThreshold=5, version=2, adaptivePN=True)
        end = time.time()
        print('SBM: ' + str(end - start))

        scatter_plot.plot_grid('SBM' + str(len(data)), data, pn, labels, marker='o', adaptivePN=True)
        plt.savefig('./figures/SBMv2_sim79_adaptive')

        pn = 25
        start = time.time()
        labels, _ = SBM_graph.SBM(spik, pn, ccThreshold=5, version=2, adaptivePN=False)
        end = time.time()
        print('SBM: ' + str(end - start))
        print(pn)
        scatter_plot.plot_grid('SBM' + str(len(data)), data, pn, labels, marker='o')
        plt.savefig('./figures/SBMv2_sim79')

        scatter_plot.plot('GT' + str(len(data)), data, y, marker='o')
        plt.savefig('./figures/ground_truth')

        unique, counts = np.unique(labels, return_counts=True)
        print(dict(zip(unique, counts)))
    elif program == "pca":
        range_min = 1
        range_max = 96
        alignment = 2
        plot_path = f'./figures/pca/'

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                continue

            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=alignment)

            # spikes = spikes[labels != 0]
            # labels = labels[labels != 0]

            pca_2d = PCA(n_components=2)
            pca_features = pca_2d.fit_transform(spikes)

            scatter_plot.plot('GT' + str(len(pca_features)), pca_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')
    elif program == "pca_selected":
        range_min = 1
        range_max = 96
        alignment = 2
        min = 30
        max = 50
        plot_path = f'./figures/pca_{min}_{max}/'

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                continue

            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=alignment)

            spikes = select_spikes(spikes, min, max)

            # spikes = spikes[labels != 0]
            # labels = labels[labels != 0]

            pca_2d = PCA(n_components=2)
            pca_features = pca_2d.fit_transform(spikes)

            scatter_plot.plot('GT' + str(len(pca_features)), pca_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

    elif program == "autoencoder":
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        simulation_number = 7
        spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)
        print(spikes.shape)

        autoencoder_layer_sizes = [70,60,50,40,30,20,10,5]
        code_size = 2
        output_activation = 'tanh'
        # output_activation = softplusplus
        # loss_function = 'mse'
        import tensorflow as tf
        # loss_function = tf.keras.losses.CategoricalCrossentropy()
        # loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        loss_function = tf.keras.losses.BinaryCrossentropy()
        nr_epochs=100
        autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                       encoder_layer_sizes=autoencoder_layer_sizes,
                                       decoder_layer_sizes=autoencoder_layer_sizes,
                                       code_size=2,
                                       output_activation=output_activation,
                                       loss_function=loss_function)
        autoencoder.train(spikes, epochs=nr_epochs)
        autoencoder.save_weights(f'./feature_extraction/autoencoder/weights/autoencoder_cs{code_size}_oa-tanh_ls-bcs_sim{simulation_number}_e{nr_epochs}')
        # autoencoder.load_weights('./feature_extraction/autoencoder/weights/autoencoder_cs{code_size}_oa-tanh_sim4_e100')
        encoder, decoder = autoencoder.return_encoder()

        verify_output(spikes, encoder, decoder, path='./feature_extraction/autoencoder/testfigs/')
        autoencoder_features = get_codes(spikes, encoder)

        print(autoencoder_features.shape)

        scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
        plt.savefig(f'./feature_extraction/autoencoder/testfigs/gt_model_cs{code_size}_oa-tanh_ls-bcs_sim{simulation_number}')

        pn = 25
        labels = SBM.best(autoencoder_features, pn, ccThreshold=5, version=2)
        scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels, marker='o')
        plt.savefig(f'./feature_extraction/autoencoder/testfigs/gt_model_cs{code_size}_oa-tanh_ls-bcs_sim{simulation_number}_sbm')

    elif program == "autoencoder_single_sim":
        range_min = 1
        range_max = 96
        epochs = 100
        learning_rate = 0.0001
        align = 2

        spike_verif_path = f'./figures/autoencoder_single_sim_epoch{epochs}_lr{learning_rate}_align{align}/spike_verif/'
        plot_path = f'./figures/autoencoder_single_sim_epoch{epochs}_lr{learning_rate}_align{align}/'
        if not os.path.exists(spike_verif_path):
            os.makedirs(spike_verif_path)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                continue

            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=align)
            unique_labels = np.unique(labels)

            autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                           encoder_layer_sizes=autoencoder_single_sim_layer_sizes,
                                           decoder_layer_sizes=autoencoder_single_sim_layer_sizes,
                                           code_size=autoencoder_single_sim_code_size)

            encoder, autoenc = autoencoder.return_encoder()

            autoencoder.train(spikes, epochs=epochs, verbose=0, learning_rate=learning_rate)
            verify_output(spikes, encoder, autoenc, i=simulation_number, path=spike_verif_path)

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            pca_features = pca_2d.fit_transform(spikes)


            scatter_plot.plot(f'GT{len(autoencoder_features)}', autoencoder_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_ae')

            scatter_plot.plot(f'GT{len(pca_features)}', pca_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca')


            pn = 25
            try:
                clustering_ae_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
                clustering_pca_labels = SBM.parallel(pca_features, pn, ccThreshold=5, version=2)

                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, clustering_ae_labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_ae_sbm')

                scatter_plot.plot_grid('SBM' + str(len(pca_features)), pca_features, pn, clustering_pca_labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca_sbm')

                hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_ae_labels)

                scores_ae = [adjusted_rand_score(labels, clustering_ae_labels),
                          adjusted_mutual_info_score(labels, clustering_ae_labels),
                          hom,
                          com,
                          vm,
                          calinski_harabasz_score(autoencoder_features, labels),
                          davies_bouldin_score(autoencoder_features, labels),
                          silhouette_score(autoencoder_features, labels)
                          ]

                print(f"{simulation_number}, {len(unique_labels)}, "
                      f"{scores_ae[0]:.2f}, {scores_ae[1]:.2f}, "
                      f"{scores_ae[2]:.2f}, {scores_ae[3]:.2f}, "
                      f"{scores_ae[4]:.2f}, {scores_ae[5]:.2f}, "
                      f"{scores_ae[6]:.2f}, {scores_ae[7]:.2f}")


                hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_pca_labels)

                scores_pca = [adjusted_rand_score(labels, clustering_pca_labels),
                          adjusted_mutual_info_score(labels, clustering_pca_labels),
                          hom,
                          com,
                          vm,
                          calinski_harabasz_score(pca_features, labels),
                          davies_bouldin_score(pca_features, labels),
                          silhouette_score(pca_features, labels)
                          ]

                print(f"{simulation_number}, {len(unique_labels)}, "
                      f"{scores_pca[0]:.2f}, {scores_pca[1]:.2f}, "
                      f"{scores_pca[2]:.2f}, {scores_pca[3]:.2f}, "
                      f"{scores_pca[4]:.2f}, {scores_pca[5]:.2f}, "
                      f"{scores_pca[6]:.2f}, {scores_pca[7]:.2f}")


                print(f"{simulation_number}, {len(unique_labels)}, "
                      f"{scores_ae[0] - scores_pca[0]:.2f}, {scores_ae[1] - scores_pca[1]:.2f}, "
                      f"{scores_ae[2] - scores_pca[2]:.2f}, {scores_ae[3] - scores_pca[3]:.2f}, "
                      f"{scores_ae[4] - scores_pca[4]:.2f}, {scores_ae[5] - scores_pca[5]:.2f}, "
                      f"{scores_ae[6] - scores_pca[6]:.2f}, {scores_ae[7] - scores_pca[7]:.2f}")

            except KeyError:
                pass

    elif program == "lstm_single_sim":
        range_min = 1
        range_max = 96
        epochs = 100
        timesteps = 20

        spike_verif_path = f'./figures/lstm_single_sim_epoch{epochs}/spike_verif/'
        plot_path = f'./figures/lstm_single_sim_epoch{epochs}/'
        if not os.path.exists(spike_verif_path):
            os.makedirs(spike_verif_path)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                continue

            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=True)
            spikes_lstm = lstm_input.temporalize_spikes(spikes, timesteps)
            unique_labels = np.unique(labels)

            autoencoder = LSTMAutoencoderModel(input_size=spikes_lstm.shape,
                                               lstm_layer_sizes=lstm_single_sim_layer_sizes,
                                               code_size=lstm_single_sim_code_size)

            encoder, autoenc = autoencoder.return_encoder()

            autoencoder.train(spikes_lstm, epochs=epochs, verbose=0)
            # verify_output(spikes, encoder, autoenc, i=simulation_number, path=spike_verif_path)

            autoencoder_features = lstm_get_codes(spikes_lstm, encoder, timesteps)

            pca_2d = PCA(n_components=2)
            pca_features = pca_2d.fit_transform(spikes)


            scatter_plot.plot(f'GT{len(autoencoder_features)}', autoencoder_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm')

            scatter_plot.plot(f'GT{len(pca_features)}', pca_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca')


            pn = 25
            try:
                clustering_ae_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
                clustering_pca_labels = SBM.parallel(pca_features, pn, ccThreshold=5, version=2)

                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, clustering_ae_labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm_sbm')

                scatter_plot.plot_grid('SBM' + str(len(pca_features)), pca_features, pn, clustering_pca_labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca_sbm')

                hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_ae_labels)

                scores_ae = [adjusted_rand_score(labels, clustering_ae_labels),
                          adjusted_mutual_info_score(labels, clustering_ae_labels),
                          hom,
                          com,
                          vm,
                          calinski_harabasz_score(autoencoder_features, labels),
                          davies_bouldin_score(autoencoder_features, labels),
                          silhouette_score(autoencoder_features, labels)
                          ]

                print(f"{simulation_number}, {len(unique_labels)}, "
                      f"{scores_ae[0]:.2f}, {scores_ae[1]:.2f}, "
                      f"{scores_ae[2]:.2f}, {scores_ae[3]:.2f}, "
                      f"{scores_ae[4]:.2f}, {scores_ae[5]:.2f}, "
                      f"{scores_ae[6]:.2f}, {scores_ae[7]:.2f}")


                hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_pca_labels)

                scores_pca = [adjusted_rand_score(labels, clustering_pca_labels),
                          adjusted_mutual_info_score(labels, clustering_pca_labels),
                          hom,
                          com,
                          vm,
                          calinski_harabasz_score(pca_features, labels),
                          davies_bouldin_score(pca_features, labels),
                          silhouette_score(pca_features, labels)
                          ]

                print(f"{simulation_number}, {len(unique_labels)}, "
                      f"{scores_pca[0]:.2f}, {scores_pca[1]:.2f}, "
                      f"{scores_pca[2]:.2f}, {scores_pca[3]:.2f}, "
                      f"{scores_pca[4]:.2f}, {scores_pca[5]:.2f}, "
                      f"{scores_pca[6]:.2f}, {scores_pca[7]:.2f}")


                print(f"{simulation_number}, {len(unique_labels)}, "
                      f"{scores_ae[0] - scores_pca[0]:.2f}, {scores_ae[1] - scores_pca[1]:.2f}, "
                      f"{scores_ae[2] - scores_pca[2]:.2f}, {scores_ae[3] - scores_pca[3]:.2f}, "
                      f"{scores_ae[4] - scores_pca[4]:.2f}, {scores_ae[5] - scores_pca[5]:.2f}, "
                      f"{scores_ae[6] - scores_pca[6]:.2f}, {scores_ae[7] - scores_pca[7]:.2f}")

            except KeyError:
                pass

    elif program == "autoencoder_sim_array":
        simulation_array = [4, 8, 79]

        spikes, labels = dsa.stack_simulations_array(simulation_array)

        from utils.constants import autoencoder_layer_sizes
        autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                       encoder_layer_sizes=autoencoder_layer_sizes,
                                       decoder_layer_sizes=autoencoder_layer_sizes,
                                       code_size=20)

        encoder, autoenc = autoencoder.return_encoder()

        if sub == "train":
            autoencoder.train(spikes, epochs=500)
            autoencoder.save_weights('./autoencoder/weights/autoencoder_array4-8-79_500')

            verify_output(spikes, encoder, autoenc)
            verify_random_outputs(spikes, encoder, autoenc, 10)
        elif sub == "test":

            autoencoder.load_weights('./autoencoder/weights/autoencoder_array4-8-79_500')

            for simulation_number in simulation_array:
                spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

                autoencoder_features = get_codes(spikes, encoder)

                scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
                plt.savefig(f'./figures/autoencoder/gt_model_sim{simulation_number}')

                pn = 25
                labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                       marker='o')
                plt.savefig(f'./figures/autoencoder/gt_model_sim{simulation_number}_sbm')
        else:
            pass

    elif program == "autoencoder_sim_range":
        range_min = 1
        range_max = 96
        epochs = 100

        autoencoder_layers = [
            [70, 60, 50, 40, 30, 20, 10],
            [70, 60, 50, 40, 30, 20],
            [70, 60, 50, 40, 30],
            [70, 60, 50, 40],
            [70, 60, 50]
        ]

        spikes, labels = dsa.stack_simulations_range(range_min, range_max, True, False)
        for autoencoder_layer_sizes in autoencoder_layers:
            plot_path = f"./figures/autoencoder_c{autoencoder_layer_sizes[-1]}_noise"
            verif_path = f"./figures/autoencoder_c{autoencoder_layer_sizes[-1]}_noise/spike_verif"
            weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_noise_c{autoencoder_layer_sizes[-1]}'

            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            if not os.path.exists(verif_path):
                os.makedirs(verif_path)

            autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                           encoder_layer_sizes=autoencoder_layer_sizes[:-1],
                                           decoder_layer_sizes=autoencoder_layer_sizes[:-1],
                                           code_size=autoencoder_layer_sizes[-1])

            encoder, autoenc = autoencoder.return_encoder()

            if sub == "train":
                autoencoder.train(spikes, epochs=epochs)
                autoencoder.save_weights(weights_path)

                verify_output(spikes, encoder, autoenc, path=verif_path)
                verify_random_outputs(spikes, encoder, autoenc, 10, path=verif_path)
            elif sub == "test":
                autoencoder.load_weights(weights_path)

                for simulation_number in range(range_min, range_max):
                    if simulation_number == 25 or simulation_number == 44:
                        continue
                    spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

                    # spikes = spikes[labels != 0]
                    # labels = labels[labels != 0]

                    autoencoder_features = get_codes(spikes, encoder)

                    pca_2d = PCA(n_components=2)
                    autoencoder_features = pca_2d.fit_transform(autoencoder_features)

                    scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
                    plt.savefig(f'{plot_path}/gt_model_sim{simulation_number}')

                    pn = 25
                    labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                    scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                           marker='o')
                    plt.savefig(f'{plot_path}/gt_model_sim{simulation_number}_sbm')
            elif sub == "pre":
                autoencoder_layer_sizes.append(autoencoder_code_size)
                layer_weights = autoencoder.pre_train(spikes, autoencoder_layer_sizes, epochs=100)
                autoencoder.set_weights(layer_weights)

                autoencoder.train(spikes, epochs=epochs)
                autoencoder.save_weights(f'./autoencoder/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_pt')
                autoencoder.load_weights(f'./autoencoder/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_pt')

                verify_output(spikes, encoder, autoenc, path=f"./figures/autoencoder_c{autoencoder_code_size}_pt/spike_verif")
                verify_random_outputs(spikes, encoder, autoenc, 10, path=f"./figures/autoencoder_c{autoencoder_code_size}_pt/spike_verif")

                for simulation_number in range(range_min, range_max):
                    if simulation_number == 25 or simulation_number == 44:
                        continue
                    spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

                    spikes = spikes[labels != 0]
                    labels = labels[labels != 0]

                    autoencoder_features = get_codes(spikes, encoder)

                    pca_2d = PCA(n_components=2)
                    autoencoder_features = pca_2d.fit_transform(autoencoder_features)

                    scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
                    plt.savefig(f'./figures/autoencoder_c{autoencoder_code_size}_pt/gt_model_sim{simulation_number}')

                    pn = 25
                    labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                    scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                           marker='o')
                    plt.savefig(f'./figures/autoencoder_c{autoencoder_code_size}_pt/gt_model_sim{simulation_number}_sbm')

            else:
                pass

    elif program == "autoencoder_selected":
        range_min = 1
        range_max = 96
        epochs = 100

        min = 20
        max = 60

        train_spikes, train_labels, test_spikes, test_labels = dsa.stack_simulations_split_train_test(range_min, range_max, no_noise=False, alignment=2, normalize=False, scale=False)

        train_spikes = select_spikes(train_spikes, min, max)
        test_selected_spikes = []
        for test in test_spikes:
            test_selected_spikes.append(select_spikes(test, min, max))

        spike_verif_path = f'./figures/autoencoder_selected2/spike_verif/'
        plot_path = f'./figures/autoencoder_selected2/'
        weights_path = f'feature_extraction/autoencoder/weights/autoencoder_selected2_allsim_e10'
        if not os.path.exists(spike_verif_path):
            os.makedirs(spike_verif_path)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        autoencoder = AutoencoderModel(input_size=len(train_spikes[0]),
                                       encoder_layer_sizes=autoencoder_selected_layer_sizes,
                                       decoder_layer_sizes=autoencoder_selected_layer_sizes,
                                       code_size=autoencoder_code_size)

        encoder, autoenc = autoencoder.return_encoder()

        if sub == "train":
            autoencoder.train(train_spikes, epochs=epochs)
            autoencoder.save_weights(weights_path)

            verify_output(train_spikes, encoder, autoenc, path=spike_verif_path)
            verify_random_outputs(train_spikes, encoder, autoenc, 10, path=spike_verif_path)

        elif sub == "test":
            autoencoder.load_weights(weights_path)

            sim_list_index = range_min
            for simulation_number in range(range_min, range_max):
                if simulation_number == 25 or simulation_number == 44:
                    continue

                spikes = test_selected_spikes[sim_list_index-1]
                labels = test_labels[sim_list_index-1]

                autoencoder_features = get_codes(spikes, encoder)

                pca_2d = PCA(n_components=2)
                autoencoder_features = pca_2d.fit_transform(autoencoder_features)

                scatter_plot.plot(f'GT{len(autoencoder_features)}/{len(train_spikes)+len(spikes)}', autoencoder_features, labels, marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

                pn = 25
                try:
                    labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                    scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                           marker='o')
                    plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
                except KeyError:
                    pass

                sim_list_index += 1

    elif program == "benchmark_autoencoder":
        autoencoder_layers = [
            [70, 60, 50, 40, 30, 20, 10],
            [70, 60, 50, 40, 30, 20],
            [70, 60, 50, 40, 30],
            [70, 60, 50, 40],
            [70, 60, 50]
        ]

        results = []
        for layers in autoencoder_layers:
            code_results = validate_model(layers[:-1], layers[-1], pt=False, noise=True)
            code_results = np.array(code_results)
            results.append(code_results)


        ari_ami_ae = []

        for i in range(len(results[0])):
            for j in range(len(autoencoder_layers)):
                ari_ami_ae.append(results[j][i])
                # print(f"{results[j][i][0]:.2f}, {results[j][i][1]:.2f}")

        results = []

        range_min = 1
        range_max = 96

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44:
                continue
            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

            # spikes = spikes[labels != 0]
            # labels = labels[labels != 0]

            pca_2d = PCA(n_components=2)
            pca_features = pca_2d.fit_transform(spikes)

            pn = 25
            clustering_labels = SBM.parallel(pca_features, pn, ccThreshold=5, version=2)

            hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_labels)

            scores = [adjusted_rand_score(labels, clustering_labels),
                      adjusted_mutual_info_score(labels, clustering_labels),
                      hom,
                      com,
                      vm,
                      calinski_harabasz_score(pca_features, labels),
                      davies_bouldin_score(pca_features, labels),
                      silhouette_score(pca_features, labels)
                      ]

            results.append(scores)

        results = np.array(results)
        ari_ami_ae = np.array(ari_ami_ae)

        for i in range(len(ari_ami_ae)):
            difference = ari_ami_ae[i] - results[i//5]
            print(f"{i//5}, {difference[0]:.2f}, {difference[1]:.2f},"
                  f"{difference[2]:.2f}, {difference[3]:.2f},"
                  f"{difference[4]:.2f}, {difference[5]:.2f},"
                  f"{difference[6]:.2f}, {difference[7]:.2f}")

        indexes = []
        nr = 5
        for i in range(0, len(ari_ami_ae) - nr, nr):
            args = np.argmax(ari_ami_ae[i:i + nr], axis=0)
            try:
                # print(statistics.mode(args), ari_ami_ae[i + statistics.mode(args)])
                indexes.append(i + statistics.mode(args))
            except statistics.StatisticsError:
                pass

        mean_best = np.mean(ari_ami_ae[indexes], axis=0)
        print(f"{mean_best[0]:.2f}, {mean_best[1]:.2f},"
              f"{mean_best[2]:.2f}, {mean_best[3]:.2f},"
              f"{mean_best[4]:.2f}, {mean_best[5]:.2f},"
              f"{mean_best[6]:.2f}, {mean_best[7]:.2f}")

        # results = np.stack(results, axis=1)
        #
        # results = np.array(results)
        # codes_ari = []
        # codes_ami = []
        #
        # for simulation_result in results:
        #     #np.argmax(simulation_result[:, 0]) - index of best code for ARI in that simulation
        #     #np.argmax(simulation_result[:, 1]) - index of best code for AMI in that simulation
        #     codes_ari.append(((np.argmax(simulation_result[:, 0]) + 1) * 10))
        #     codes_ami.append(((np.argmax(simulation_result[:, 1]) + 1) * 10))
        #
        # codes, counts = np.unique(codes_ari, return_counts=True)
        # codes_n_counts_ari = dict(zip(codes, counts))
        # codes, counts = np.unique(codes_ami, return_counts=True)
        # codes_n_counts_ami = dict(zip(codes, counts))
        #
        # for i in range(0, len(autoencoder_layers)):
        #     ari_ami_values_for_all_sim = results[:, i]
        #     mean_values = np.mean(ari_ami_values_for_all_sim, axis=0)
        #     print(f"CODE {(i+1)*10} -> ARI - {mean_values[0]}")
        #     print(f"CODE {(i+1)*10} -> AMI - {mean_values[1]}")
        #
        #
        # print(codes_n_counts_ari)
        # print(codes_n_counts_ami)

        # np.savetxt("./autoencoder/test.csv", results, delimiter=",", fmt='%.2f')

    elif program == "benchmark_pca":
        results = []

        range_min = 1
        range_max = 96

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44:
                continue
            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

            uniq_labels = np.unique(labels)

            # spikes = spikes[labels != 0]
            # labels = labels[labels != 0]

            pca_2d = PCA(n_components=2)
            pca_features = pca_2d.fit_transform(spikes)

            pn = 25
            clustering_labels = SBM.parallel(pca_features, pn, ccThreshold=5, version=2)

            hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_labels)

            scores = [adjusted_rand_score(labels, clustering_labels),
                            adjusted_mutual_info_score(labels, clustering_labels),
                            hom,
                            com,
                            vm,
                            calinski_harabasz_score(pca_features, labels),
                            davies_bouldin_score(pca_features, labels),
                            silhouette_score(pca_features, labels)
                            ]

            results.append(scores)

            print(f"{scores[0]:.2f}, {scores[1]:.2f}, "
                  f"{scores[2]:.2f}, {scores[3]:.2f}, "
                  f"{scores[4]:.2f}, {scores[5]:.2f}, "
                  f"{scores[6]:.2f}, {scores[7]:.2f}")


        results = np.array(results)

        mean_values = np.mean(results, axis=0)

        print(f"PCA -> ARI - {mean_values[0]}")
        print(f"PCA -> AMI - {mean_values[1]}")
        print(f"PCA -> Hom - {mean_values[2]}")
        print(f"PCA -> Com - {mean_values[3]}")
        print(f"PCA -> VM - {mean_values[4]}")
        print(f"PCA -> CHS - {mean_values[5]}")
        print(f"PCA -> DBS - {mean_values[6]}")
        print(f"PCA -> SS - {mean_values[7]}")

    elif program == "benchmark_lstm":
        range_min = 1
        range_max = 96
        epochs = 100
        timesteps = 20
        overlap = 0
        no_noise = True
        align = True
        normalize = False
        scale = False

        # plot_path = f'./figures/lstm_verif_c{lstm_code_size}_TS{timesteps}_OL{overlap}/'
        plot_path = f'./figures/lstm_verif_c{lstm_code_size}_TS{timesteps}_OL{overlap}_nonoise/'
        # weights_path = f'./autoencoder/weights/lstm_allsim_e100_d80_nonoise_c{lstm_code_size}_TS{timesteps}_OL{overlap}'
        weights_path = f'feature_extraction/autoencoder/weights/lstm_allsim_e100_d80_nonoise_c{lstm_code_size}_TS{timesteps}_OL{overlap}_nonoise_norm{normalize}_scale_{scale}'

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        avg_pca = []
        avg_lstm = []


        train_spikes, train_labels, test_spikes, test_labels = dsa.stack_simulations_split_train_test(range_min, range_max, no_noise=no_noise, alignment=align, normalize=normalize, scale=scale)
        train_spikes = lstm_input.temporalize_spikes(train_spikes, timesteps, overlap)

        autoencoder = LSTMAutoencoderModel(input_size=train_spikes.shape,
                                           lstm_layer_sizes=lstm_layer_sizes,
                                           code_size=lstm_code_size)

        encoder, autoenc = autoencoder.return_encoder()

        autoencoder.load_weights(weights_path)

        results_pca = []
        results_lstm = []
        sim_list_index = range_min
        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44:
                continue

            spikes = test_spikes[sim_list_index - 1]
            labels = test_labels[sim_list_index - 1]

            pca_2d = PCA(n_components=2)
            pca_features = pca_2d.fit_transform(spikes)

            spikes = lstm_input.temporalize_spikes(spikes, timesteps, overlap)

            autoencoder_features = lstm_get_codes(spikes, encoder, timesteps)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)


            scatter_plot.plot(f'GT{len(autoencoder_features)}', autoencoder_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm')

            scatter_plot.plot(f'GT{len(pca_features)}', pca_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca')


            pn = 25
            try:
                clustering_lstm_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
                clustering_pca_labels = SBM.parallel(pca_features, pn, ccThreshold=5, version=2)

                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, clustering_lstm_labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm_sbm')

                scatter_plot.plot_grid('SBM' + str(len(pca_features)), pca_features, pn, clustering_pca_labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca_sbm')

                hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_lstm_labels)

                scores_lstm = [adjusted_rand_score(labels, clustering_lstm_labels),
                          adjusted_mutual_info_score(labels, clustering_lstm_labels),
                          hom,
                          com,
                          vm,
                          calinski_harabasz_score(autoencoder_features, labels),
                          davies_bouldin_score(autoencoder_features, labels),
                          silhouette_score(autoencoder_features, labels)
                          ]

                results_lstm.append(scores_lstm)

                print(f"{sim_list_index}, {scores_lstm[0]:.2f}, {scores_lstm[1]:.2f}, "
                      f"{scores_lstm[2]:.2f}, {scores_lstm[3]:.2f}, "
                      f"{scores_lstm[4]:.2f}, {scores_lstm[5]:.2f}, "
                      f"{scores_lstm[6]:.2f}, {scores_lstm[7]:.2f}")


                hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_pca_labels)

                scores_pca = [adjusted_rand_score(labels, clustering_pca_labels),
                          adjusted_mutual_info_score(labels, clustering_pca_labels),
                          hom,
                          com,
                          vm,
                          calinski_harabasz_score(pca_features, labels),
                          davies_bouldin_score(pca_features, labels),
                          silhouette_score(pca_features, labels)
                          ]

                print(f"{sim_list_index}, {scores_pca[0]:.2f}, {scores_pca[1]:.2f}, "
                      f"{scores_pca[2]:.2f}, {scores_pca[3]:.2f}, "
                      f"{scores_pca[4]:.2f}, {scores_pca[5]:.2f}, "
                      f"{scores_pca[6]:.2f}, {scores_pca[7]:.2f}")


                print(f"{sim_list_index}, {scores_lstm[0] - scores_pca[0]:.2f}, {scores_lstm[1] - scores_pca[1]:.2f}, "
                      f"{scores_lstm[2] - scores_pca[2]:.2f}, {scores_lstm[3] - scores_pca[3]:.2f}, "
                      f"{scores_lstm[4] - scores_pca[4]:.2f}, {scores_lstm[5] - scores_pca[5]:.2f}, "
                      f"{scores_lstm[6] - scores_pca[6]:.2f}, {scores_lstm[7] - scores_pca[7]:.2f}")

                results_pca.append(scores_pca)



            except KeyError:
                pass

            sim_list_index += 1

        results_lstm = np.array(results_lstm)
        results_pca = np.array(results_pca)

        mean_values = np.mean(results_lstm, axis=0)

        print(f"LSTM -> ARI - {mean_values[0]}")
        print(f"LSTM -> AMI - {mean_values[1]}")
        print(f"LSTM -> Hom - {mean_values[2]}")
        print(f"LSTM -> Com - {mean_values[3]}")
        print(f"LSTM -> VM - {mean_values[4]}")
        print(f"LSTM -> CHS - {mean_values[5]}")
        print(f"LSTM -> DBS - {mean_values[6]}")
        print(f"LSTM -> SS - {mean_values[7]}")

        mean_values = np.mean(results_pca, axis=0)

        print(f"PCA -> ARI - {mean_values[0]}")
        print(f"PCA -> AMI - {mean_values[1]}")
        print(f"PCA -> Hom - {mean_values[2]}")
        print(f"PCA -> Com - {mean_values[3]}")
        print(f"PCA -> VM - {mean_values[4]}")
        print(f"PCA -> CHS - {mean_values[5]}")
        print(f"PCA -> DBS - {mean_values[6]}")
        print(f"PCA -> SS - {mean_values[7]}")

    elif program == "benchmark_lstm_mul":
        range_min = 1
        range_max = 96
        epochs = 100
        timesteps = 20
        overlap = 0
        no_noise = True
        align = True
        normalize = False
        scale = False

        # plot_path = f'./figures/lstm_verif_c{lstm_code_size}_TS{timesteps}_OL{overlap}/'
        plot_path = f'./figures/lstm_verif_c{lstm_code_size}_TS{timesteps}_OL{overlap}_nonoise/'
        # weights_path = f'./autoencoder/weights/lstm_allsim_e100_d80_nonoise_c{lstm_code_size}_TS{timesteps}_OL{overlap}'
        weights_path = f'feature_extraction/autoencoder/weights/lstm_allsim_e100_d80_nonoise_c{lstm_code_size}_TS{timesteps}_OL{overlap}_nonoise_norm{normalize}_scale_{scale}'

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        avg_pca = []
        avg_lstm = []

        train_spikes, train_labels, test_spikes, test_labels = dsa.stack_simulations_split_train_test(range_min, range_max, no_noise=no_noise, alignment=align, normalize=normalize, scale=scale)
        train_spikes = lstm_input.temporalize_spikes(train_spikes, timesteps, overlap)

        autoencoder = LSTMAutoencoderModel(input_size=train_spikes.shape,
                                           lstm_layer_sizes=lstm_layer_sizes,
                                           code_size=lstm_code_size)

        encoder, autoenc = autoencoder.return_encoder()

        autoencoder.load_weights(weights_path)

        results_pca = []
        results_lstm = []
        sim_list_index = range_min
        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44:
                continue

            spikes = test_spikes[sim_list_index - 1]
            labels = test_labels[sim_list_index - 1]

            simulation_pca = []
            simulation_lstm = []
            for test in range(0, 10):
                pca_2d = PCA(n_components=2)
                pca_features = pca_2d.fit_transform(spikes)

                spikes_lstm = lstm_input.temporalize_spikes(spikes, timesteps, overlap)

                autoencoder_features = lstm_get_codes(spikes_lstm, encoder, timesteps)

                pca_2d = PCA(n_components=2)
                autoencoder_features = pca_2d.fit_transform(autoencoder_features)

                # scatter_plot.plot(f'GT{len(autoencoder_features)}', autoencoder_features, labels, marker='o')
                # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm')
                #
                # scatter_plot.plot(f'GT{len(pca_features)}', pca_features, labels, marker='o')
                # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca')

                pn = 25
                try:
                    clustering_lstm_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
                    clustering_pca_labels = SBM.parallel(pca_features, pn, ccThreshold=5, version=2)

                    # scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, clustering_lstm_labels,
                    #                        marker='o')
                    # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm_sbm')
                    #
                    # scatter_plot.plot_grid('SBM' + str(len(pca_features)), pca_features, pn, clustering_pca_labels,
                    #                        marker='o')
                    # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca_sbm')

                    hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_lstm_labels)

                    scores_lstm = [adjusted_rand_score(labels, clustering_lstm_labels),
                                   adjusted_mutual_info_score(labels, clustering_lstm_labels),
                                   hom,
                                   com,
                                   vm,
                                   calinski_harabasz_score(autoencoder_features, labels),
                                   davies_bouldin_score(autoencoder_features, labels),
                                   silhouette_score(autoencoder_features, labels)
                                   ]

                    simulation_lstm.append(scores_lstm)

                    hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_pca_labels)

                    scores_pca = [adjusted_rand_score(labels, clustering_pca_labels),
                                  adjusted_mutual_info_score(labels, clustering_pca_labels),
                                  hom,
                                  com,
                                  vm,
                                  calinski_harabasz_score(pca_features, labels),
                                  davies_bouldin_score(pca_features, labels),
                                  silhouette_score(pca_features, labels)
                                  ]

                    simulation_pca.append(scores_pca)

                except KeyError:
                    pass

            sim_list_index += 1

            simulation_lstm = np.array(simulation_lstm)
            simulation_pca = np.array(simulation_pca)

            mean_lstm = np.mean(simulation_lstm, axis=0)
            mean_pca = np.mean(simulation_pca, axis=0)

            print(f"{sim_list_index}, {mean_lstm[0]:.2f}, {mean_lstm[1]:.2f}, "
                  f"{mean_lstm[2]:.2f}, {mean_lstm[3]:.2f}, "
                  f"{mean_lstm[4]:.2f}, {mean_lstm[5]:.2f}, "
                  f"{mean_lstm[6]:.2f}, {mean_lstm[7]:.2f}")

            results_lstm.append(mean_lstm)

            print(f"{sim_list_index}, {mean_pca[0]:.2f}, {mean_pca[1]:.2f}, "
                  f"{mean_pca[2]:.2f}, {mean_pca[3]:.2f}, "
                  f"{mean_pca[4]:.2f}, {mean_pca[5]:.2f}, "
                  f"{mean_pca[6]:.2f}, {mean_pca[7]:.2f}")

            print(f"{sim_list_index}, {mean_lstm[0] - mean_pca[0]:.2f}, {mean_lstm[1] - mean_pca[1]:.2f}, "
                  f"{mean_lstm[2] - mean_pca[2]:.2f}, {mean_lstm[3] - mean_pca[3]:.2f}, "
                  f"{mean_lstm[4] - mean_pca[4]:.2f}, {mean_lstm[5] - mean_pca[5]:.2f}, "
                  f"{mean_lstm[6] - mean_pca[6]:.2f}, {mean_lstm[7] - mean_pca[7]:.2f}")

            results_pca.append(mean_pca)

        results_lstm = np.array(results_lstm)
        results_pca = np.array(results_pca)

        mean_values = np.mean(results_lstm, axis=0)

        print(f"LSTM -> ARI - {mean_values[0]}")
        print(f"LSTM -> AMI - {mean_values[1]}")
        print(f"LSTM -> Hom - {mean_values[2]}")
        print(f"LSTM -> Com - {mean_values[3]}")
        print(f"LSTM -> VM - {mean_values[4]}")
        print(f"LSTM -> CHS - {mean_values[5]}")
        print(f"LSTM -> DBS - {mean_values[6]}")
        print(f"LSTM -> SS - {mean_values[7]}")

        mean_values = np.mean(results_pca, axis=0)

        print(f"PCA -> ARI - {mean_values[0]}")
        print(f"PCA -> AMI - {mean_values[1]}")
        print(f"PCA -> Hom - {mean_values[2]}")
        print(f"PCA -> Com - {mean_values[3]}")
        print(f"PCA -> VM - {mean_values[4]}")
        print(f"PCA -> CHS - {mean_values[5]}")
        print(f"PCA -> DBS - {mean_values[6]}")
        print(f"PCA -> SS - {mean_values[7]}")

    elif program == "benchmark_lstm_multiple":
        range_min = 1
        range_max = 96
        epochs = 100
        timesteps = 20
        overlap = 0
        no_noise = False
        align = True
        normalize = False
        scale = False

        plot_path = f'./figures/lstm_verif_c{lstm_code_size}_TS{timesteps}_OL{overlap}/'
        weights_path = f'feature_extraction/autoencoder/weights/lstm_allsim_e100_d80_nonoise_c{lstm_code_size}_TS{timesteps}_OL{overlap}'

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        avg_pca = []
        avg_lstm = []
        for test in range(0, 10):
            train_spikes, train_labels, test_spikes, test_labels = dsa.stack_simulations_split_train_test(range_min, range_max, no_noise=False, alignment=align, normalize=normalize, scale=scale)
            train_spikes = lstm_input.temporalize_spikes(train_spikes, timesteps, overlap)

            autoencoder = LSTMAutoencoderModel(input_size=train_spikes.shape,
                                               lstm_layer_sizes=lstm_layer_sizes,
                                               code_size=lstm_code_size)

            encoder, autoenc = autoencoder.return_encoder()

            autoencoder.load_weights(weights_path)

            results_pca = []
            results_lstm = []
            sim_list_index = range_min
            for simulation_number in range(range_min, range_max):
                if simulation_number == 25 or simulation_number == 44:
                    continue

                spikes = test_spikes[sim_list_index - 1]
                labels = test_labels[sim_list_index - 1]

                pca_2d = PCA(n_components=2)
                pca_features = pca_2d.fit_transform(spikes)

                spikes = lstm_input.temporalize_spikes(spikes, timesteps, overlap)

                autoencoder_features = lstm_get_codes(spikes, encoder, timesteps)

                pca_2d = PCA(n_components=2)
                autoencoder_features = pca_2d.fit_transform(autoencoder_features)


                # scatter_plot.plot(f'GT{len(autoencoder_features)}', autoencoder_features, labels, marker='o')
                # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm')
                #
                # scatter_plot.plot(f'GT{len(pca_features)}', pca_features, labels, marker='o')
                # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca')


                pn = 25
                try:
                    clustering_lstm_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
                    clustering_pca_labels = SBM.parallel(pca_features, pn, ccThreshold=5, version=2)

                    # scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, clustering_lstm_labels,
                    #                        marker='o')
                    # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm_sbm')
                    #
                    # scatter_plot.plot_grid('SBM' + str(len(pca_features)), pca_features, pn, clustering_pca_labels,
                    #                        marker='o')
                    # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca_sbm')

                    hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_lstm_labels)

                    scores = [adjusted_rand_score(labels, clustering_lstm_labels),
                              adjusted_mutual_info_score(labels, clustering_lstm_labels),
                              hom,
                              com,
                              vm,
                              calinski_harabasz_score(autoencoder_features, labels),
                              davies_bouldin_score(autoencoder_features, labels),
                              silhouette_score(autoencoder_features, labels)
                              ]

                    results_lstm.append(scores)

                    print(f"{scores[0]:.2f}, {scores[1]:.2f}, "
                          f"{scores[2]:.2f}, {scores[3]:.2f}, "
                          f"{scores[4]:.2f}, {scores[5]:.2f}, "
                          f"{scores[6]:.2f}, {scores[7]:.2f}")


                    hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_pca_labels)

                    scores = [adjusted_rand_score(labels, clustering_pca_labels),
                              adjusted_mutual_info_score(labels, clustering_pca_labels),
                              hom,
                              com,
                              vm,
                              calinski_harabasz_score(pca_features, labels),
                              davies_bouldin_score(pca_features, labels),
                              silhouette_score(pca_features, labels)
                              ]

                    print(f"{scores[0]:.2f}, {scores[1]:.2f}, "
                          f"{scores[2]:.2f}, {scores[3]:.2f}, "
                          f"{scores[4]:.2f}, {scores[5]:.2f}, "
                          f"{scores[6]:.2f}, {scores[7]:.2f}")

                    results_pca.append(scores)

                    sim_list_index += 1

                except KeyError:
                    pass
            results_lstm = np.array(results_lstm)
            results_pca = np.array(results_pca)

            mean_values = np.mean(results_lstm, axis=0)
            avg_lstm.append(mean_values)
            mean_values = np.mean(results_pca, axis=0)
            avg_pca.append(mean_values)

        avg_lstm = np.array(avg_lstm)
        avg_pca = np.array(avg_pca)

        mean_values = np.mean(avg_lstm, axis=0)

        print(f"LSTM -> ARI - {mean_values[0]}")
        print(f"LSTM -> AMI - {mean_values[1]}")
        print(f"LSTM -> Hom - {mean_values[2]}")
        print(f"LSTM -> Com - {mean_values[3]}")
        print(f"LSTM -> VM - {mean_values[4]}")
        print(f"LSTM -> CHS - {mean_values[5]}")
        print(f"LSTM -> DBS - {mean_values[6]}")
        print(f"LSTM -> SS - {mean_values[7]}")

        mean_values = np.mean(avg_pca, axis=0)

        print(f"PCA -> ARI - {mean_values[0]}")
        print(f"PCA -> AMI - {mean_values[1]}")
        print(f"PCA -> Hom - {mean_values[2]}")
        print(f"PCA -> Com - {mean_values[3]}")
        print(f"PCA -> VM - {mean_values[4]}")
        print(f"PCA -> CHS - {mean_values[5]}")
        print(f"PCA -> DBS - {mean_values[6]}")
        print(f"PCA -> SS - {mean_values[7]}")

    elif program == "pipeline_test":
        range_min = 1
        range_max = 96
        autoencoder_layers = [70, 60, 50, 40, 30, 20]

        autoencoder = AutoencoderModel(input_size=79,
                                       encoder_layer_sizes=autoencoder_layers[:-1],
                                       decoder_layer_sizes=autoencoder_layers[:-1],
                                       code_size=autoencoder_layers[-1])

        encoder, autoenc = autoencoder.return_encoder()

        autoencoder.load_weights(f'./autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c20')

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44:
                continue
            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

            spikes = spikes[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            sil_coeffs = metrics.silhouette_samples(autoencoder_features, labels, metric='mahalanobis')
            means = []
            for label in np.arange(max(labels) + 1):
                if label not in labels:
                    means.append(-1)
                else:
                    means.append(sil_coeffs[labels == label].mean())
            for label in np.arange(max(labels) + 1):
                if means[label] > 0.7:
                    print(f"SIM{simulation_number} separates {label}")

    elif program == "lstm":
        range_min = 1
        range_max = 96
        epochs = 100
        timesteps = 20
        overlap = 0
        nonoise = True
        align = 2

        spike_verif_path = f'./figures/lstm_c{lstm_code_size}_nn{nonoise}_align{align}/spike_verif/'
        plot_path = f'./figures/lstm_c{lstm_code_size}_nn{nonoise}_align{align}/'
        weights_path = f'feature_extraction/autoencoder/weights/lstm_allsim_e100_d80_nonoise_c{lstm_code_size}_nn{nonoise}_align{align}'
        if not os.path.exists(spike_verif_path):
            os.makedirs(spike_verif_path)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        # spikes, labels = ds.stack_simulations_range(range_min, range_max, True, True)
        # spikes = np.reshape(spikes, (spikes.shape[0], spikes.shape[1], 1))

        spikes, labels = dsa.stack_simulations_range(range_min, range_max, False, nonoise, align)
        # spikes, labels = lstm_input.temporalize_spikes(spikes, labels, timesteps)
        spikes = lstm_input.temporalize_spikes(spikes, timesteps)

        autoencoder = LSTMAutoencoderModel(input_size=spikes.shape,
                                       lstm_layer_sizes=lstm_layer_sizes,
                                       code_size=lstm_code_size)

        encoder, autoenc = autoencoder.return_encoder()

        if sub == "train":
            autoencoder.train(spikes, epochs=epochs)
            autoencoder.save_weights(weights_path)

            # verify_output(spikes, encoder, autoenc, path=spike_verif_path)
            # verify_random_outputs(spikes, encoder, autoenc, 10, path=spike_verif_path)
        elif sub == "test":
            autoencoder.load_weights(weights_path)

            for simulation_number in range(range_min, range_max):
                if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                    continue
                spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

                spikes = spikes[labels!=0]
                labels = labels[labels!=0]

                # spikes, labels = lstm_input.temporalize_data(spikes, labels, timesteps)
                spikes = lstm_input.temporalize_spikes(spikes, timesteps)

                autoencoder_features = lstm_get_codes(spikes, encoder, timesteps)

                pca_2d = PCA(n_components=2)
                autoencoder_features = pca_2d.fit_transform(autoencoder_features)

                scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

                pn = 25
                labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
        elif sub == "validation":
            autoencoder.load_weights(weights_path)

            for simulation_number in range(range_min, range_max):
                if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                    continue
                spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)
                unique_labels = np.unique(labels)

                spikes = spikes[labels!=0]
                labels = labels[labels!=0]
                # spikes, labels = lstm_input.temporalize_data(spikes, labels, timesteps)
                spikes = lstm_input.temporalize_spikes(spikes, timesteps)

                autoencoder_features = lstm_get_codes(spikes, encoder, timesteps)

                pca_2d = PCA(n_components=2)
                autoencoder_features = pca_2d.fit_transform(autoencoder_features)

                clustering_ae_labels = SBM.parallel(autoencoder_features, 25, ccThreshold=5, version=2)

                hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_ae_labels)

                scores_ae = [adjusted_rand_score(labels, clustering_ae_labels),
                             adjusted_mutual_info_score(labels, clustering_ae_labels),
                             hom,
                             com,
                             vm,
                             calinski_harabasz_score(autoencoder_features, labels),
                             davies_bouldin_score(autoencoder_features, labels),
                             silhouette_score(autoencoder_features, labels)
                             ]

                print(f"{simulation_number}, {len(unique_labels)}, "
                      f"{scores_ae[0]:.2f}, {scores_ae[1]:.2f}, "
                      f"{scores_ae[2]:.2f}, {scores_ae[3]:.2f}, "
                      f"{scores_ae[4]:.2f}, {scores_ae[5]:.2f}, "
                      f"{scores_ae[6]:.2f}, {scores_ae[7]:.2f}")

    elif program == "split_lstm":
        range_min = 1
        range_max = 96
        epochs = 100
        timesteps = 20
        overlap = 0
        no_noise = True
        align = True
        normalize = False
        scale = False

        spike_verif_path = f'./figures/lstm_c{lstm_code_size}_TS{timesteps}_OL{overlap}_norm{normalize}_scale_{scale}_nonoise/spike_verif/'
        plot_path = f'./figures/lstm_c{lstm_code_size}_TS{timesteps}_OL{overlap}_norm{normalize}_scale_{scale}_nonoise/'
        weights_path = f'feature_extraction/autoencoder/weights/lstm_allsim_e100_d80_nonoise_c{lstm_code_size}_TS{timesteps}_OL{overlap}_nonoise_norm{normalize}_scale_{scale}'
        if not os.path.exists(spike_verif_path):
            os.makedirs(spike_verif_path)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        train_spikes, train_labels, test_spikes, test_labels = dsa.stack_simulations_split_train_test(range_min, range_max, no_noise=no_noise, alignment=align, normalize=normalize, scale=scale)
        train_spikes = lstm_input.temporalize_spikes(train_spikes, timesteps, overlap)

        autoencoder = LSTMAutoencoderModel(input_size=train_spikes.shape,
                                           lstm_layer_sizes=lstm_layer_sizes,
                                           code_size=lstm_code_size)

        encoder, autoenc = autoencoder.return_encoder()

        if sub == "train":
            autoencoder.train(train_spikes, epochs=epochs)
            autoencoder.save_weights(weights_path)

            lstm_input.lstm_verify_output(train_spikes, timesteps, encoder, autoenc, path=spike_verif_path)
            lstm_input.lstm_verify_random_outputs(train_spikes, timesteps, encoder, autoenc, 10, path=spike_verif_path)
        elif sub == "test":
            autoencoder.load_weights(weights_path)

            sim_list_index = range_min
            for simulation_number in range(range_min, range_max):
                if simulation_number == 25 or simulation_number == 44:
                    continue

                spikes = test_spikes[sim_list_index-1]
                labels = test_labels[sim_list_index-1]

                spikes = lstm_input.temporalize_spikes(spikes, timesteps, overlap)

                autoencoder_features = lstm_get_codes(spikes, encoder, timesteps)

                pca_2d = PCA(n_components=2)
                autoencoder_features = pca_2d.fit_transform(autoencoder_features)

                scatter_plot.plot(f'GT{len(autoencoder_features)}/{len(train_spikes)+len(spikes)}', autoencoder_features, labels, marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

                pn = 25
                try:
                    labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                    scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                           marker='o')
                    plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
                except KeyError:
                    pass

                sim_list_index += 1

    elif program == "lstm_pca_check":
        plot_path = f'./figures/lstm_norm_pca_check/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        for simulation_number in range(1, 96):
            if simulation_number == 25 or simulation_number == 44:
                continue
            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=True, normalize_spike=True)

            check_spikes = spikes[:, 20:40]

            pca_2d = PCA(n_components=2)
            new_features = pca_2d.fit_transform(check_spikes)

            scatter_plot.plot(f'GT{len(new_features)}', new_features,
                              labels, marker='o')
            plt.savefig(plot_path + f'check_sim{simulation_number}')

    elif program == "sbm_graph_merge":
        simulation_number = 4
        data, y = ds.get_dataset_simulation(simNr=simulation_number)

        # dims = 2
        for dims in range(2, 7):
            print(dims)
            pca_2d = PCA(n_components=dims)
            spikes = pca_2d.fit_transform(data)

            # scatter_plot.plot('GT-' + str(len(spikes)), spikes, labels, marker='o')
            # plt.savefig('./figures/sim4_gt')

            pn = 25
            start = time.time()
            labels_og = SBM.best(spikes, pn)
            sbmog_time = time.time() - start
            print(f"---> SBM-og: {sbmog_time:.3f} with {(len(labels_og)-np.count_nonzero(labels_og))/len(labels_og)*100}")


            start = time.time()
            labels = SBM_graph_merge.SBM(spikes, pn)
            sbmv2_time = time.time() - start
            print(f"---> SBM-v2: {sbmv2_time:.3f} with {(len(labels)-np.count_nonzero(labels))/len(labels)*100}")

            # sum_time = 0
            # nr = 20
            # for test in range(nr):
            #     start = time.time()
            #     labels = SBM_graph.SBM(spikes, pn)
            #     sum_time += time.time() - start
            # print(f"SBMv2: {sum_time/nr}")

            pca_2d = PCA(n_components=2)
            spikes = pca_2d.fit_transform(data)

            scatter_plot.plot('GT-' + str(len(spikes)), spikes, y, marker='o')
            plt.savefig(f'./figures/sim4_gt_dim{dims}')
            scatter_plot.plot_grid(f'SBMog {len(spikes)} in {sbmog_time:.3f}', spikes, pn, labels_og, marker='o')
            plt.savefig(f'./figures/sim4_sbm_og_dim{dims}')
            scatter_plot.plot_grid(f'SBMv2 {len(spikes)} in {sbmv2_time:.3f}', spikes, pn, labels, marker='o')
            plt.savefig(f'./figures/sim4_sbm_v2_merged_dim{dims}')

            print()

    elif program == "show_spike_shapes":
        range_min = 1
        range_max = 96

        plot_path = f'./figures/spike_shapes/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                continue
            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

            folder_path = f'./figures/spike_shapes/sim{simulation_number}/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            unique_labels = np.unique(labels)

            for label in unique_labels:
                cluster_spikes = spikes[labels == label]

                plt.figure()
                for i in range(0, len(cluster_spikes)):
                    plt.plot(np.arange(len(cluster_spikes[i])), cluster_spikes[i])
                plt.xlabel('Time')
                plt.ylabel('Magnitude')
                plt.title(f"Spikes of cluster {label}")
                plt.savefig(f'./figures/spike_shapes/sim{simulation_number}/' + f"cluster{label}")


def select_spikes(spikes, range_min, range_max):
    spikes_list = []

    for spike in spikes:
        spikes_list.append(spike[range_min:range_max])

    return np.array(spikes_list)


def main_ensemble_stacked():
    range_min = 1
    range_max = 96
    epochs = 100
    on_type = "magnitude"
    align = 2
    normalize = True
    scale = False
    plot_path = f'./figures/fft_windowed_ensemble_stacking/'

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)


    spikes_list, labels_list = dsa.split_stack_simulations(range_min, range_max, no_noise=False, alignment=align, normalize=normalize, scale=scale)

    for i in range(0, len(spikes_list), 2):
        fft_real_blackman, fft_imag_blackman = fft_input.apply_fft_windowed_on_data(spikes_list[i], "blackman")
        fft_real_gaussian, fft_imag_gaussian = fft_input.apply_fft_windowed_on_data(spikes_list[i + 1], "gaussian")

        spikes_blackman = get_type(on_type, fft_real_blackman, fft_imag_blackman)
        spikes_list[i] = np.array(spikes_blackman)
        spikes_gaussian = get_type(on_type, fft_real_gaussian, fft_imag_gaussian)
        spikes_list[i+1] = np.array(spikes_gaussian)

    model_list = []

    for i in range(0, len(spikes_list)-2, 2):
        model_list.append(AutoencoderModel(input_size=len(spikes_list[i][0]),
                                       encoder_layer_sizes=autoencoder_layer_sizes,
                                       decoder_layer_sizes=autoencoder_layer_sizes,
                                       code_size=autoencoder_code_size))

        model_list.append(AutoencoderModel(input_size=len(spikes_list[i+1][0]),
                                       encoder_layer_sizes=autoencoder_layer_sizes,
                                       decoder_layer_sizes=autoencoder_layer_sizes,
                                       code_size=autoencoder_code_size))

    encoder_list = []
    autoencoder_list = []
    for i in range(len(model_list)):
        encoder, autoenc = model_list[i].return_encoder()
        encoder_list.append(encoder)
        autoencoder_list.append(autoenc)

    for i in range(len(model_list)):
        model_list[i].train(spikes_list[i], epochs=epochs)

    outputs_list = []
    for i in range(len(autoencoder_list)):
        outputs_list.append(np.empty((128,)))
        for j in range(len(spikes_list[i])):
            decoded_spike = autoencoder_list[i].predict(spikes_list[i][j].reshape(1, -1))
            outputs_list[i] = np.vstack((outputs_list[i], decoded_spike[0]))
        outputs_list[i] = outputs_list[i][1:]

    output_spikes = np.empty((128,))
    for i in range(len(outputs_list)):
        output_spikes = np.vstack((output_spikes, outputs_list[i]))

    output_spikes = np.vstack((output_spikes, spikes_list[-2]))
    output_spikes = np.vstack((output_spikes, spikes_list[-1]))
    output_spikes = output_spikes[1:]

    final_autoencoder = AutoencoderModel(input_size=len(output_spikes[0]),
                     encoder_layer_sizes=autoencoder_layer_sizes,
                     decoder_layer_sizes=autoencoder_layer_sizes,
                     code_size=autoencoder_code_size)
    encoder, autoenc = final_autoencoder.return_encoder()
    final_autoencoder.train(output_spikes, epochs=epochs)

    results_ensemble_blackman = []
    results_ensemble_gaussian = []
    for simulation_number in range(range_min, range_max):
        if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
            continue

        fft_real_blackman, fft_imag_blackman, labels_blackman = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                                                    alignment=align,
                                                                                                    window_type="blackman")
        fft_real_gaussian, fft_imag_gaussian, labels_gaussian = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                                                    alignment=align,
                                                                                                    window_type="gaussian")

        fft_real_blackman = np.array(fft_real_blackman)
        fft_imag_blackman = np.array(fft_imag_blackman)
        fft_real_gaussian = np.array(fft_real_gaussian)
        fft_imag_gaussian = np.array(fft_imag_gaussian)

        spikes_blackman = get_type(on_type, fft_real_blackman, fft_imag_blackman)
        spikes_blackman = np.array(spikes_blackman)
        spikes_gaussian = get_type(on_type, fft_real_gaussian, fft_imag_gaussian)
        spikes_gaussian = np.array(spikes_gaussian)

        spikes_gaussian = spikes_gaussian[labels_gaussian != 0]
        labels_gaussian = labels_gaussian[labels_gaussian != 0]
        spikes_blackman = spikes_blackman[labels_blackman != 0]
        labels_blackman = labels_blackman[labels_blackman != 0]

        autoencoder_features_blackman = get_codes(spikes_gaussian, encoder)
        autoencoder_features_gaussian = get_codes(spikes_blackman, encoder)

        pca_2d = PCA(n_components=2)
        autoencoder_features_blackman = pca_2d.fit_transform(autoencoder_features_blackman)
        autoencoder_features_gaussian = pca_2d.fit_transform(autoencoder_features_gaussian)

        scatter_plot.plot('GT' + str(len(autoencoder_features_blackman)), autoencoder_features_blackman, labels_blackman, marker='o')
        plt.savefig(plot_path + f'gt_sim{simulation_number}_gaussian')
        scatter_plot.plot('GT' + str(len(autoencoder_features_gaussian)), autoencoder_features_gaussian, labels_gaussian, marker='o')
        plt.savefig(plot_path + f'gt_sim{simulation_number}_blackman')

        clustering_labels = SBM.best(autoencoder_features_blackman, 25, ccThreshold=5, version=2)
        sil_coeffs1 = silhouette_samples(autoencoder_features_blackman, labels_blackman, metric='mahalanobis')

        results_ensemble_blackman.append([adjusted_rand_score(labels_blackman, clustering_labels),
                                 adjusted_mutual_info_score(labels_blackman, clustering_labels),
                                 np.average(sil_coeffs1)])

        clustering_labels = SBM.best(autoencoder_features_gaussian, 25, ccThreshold=5, version=2)
        sil_coeffs2 = silhouette_samples(autoencoder_features_gaussian, labels_gaussian, metric='mahalanobis')


        results_ensemble_gaussian.append([adjusted_rand_score(labels_gaussian, clustering_labels),
                                 adjusted_mutual_info_score(labels_gaussian, clustering_labels),
                                 np.average(sil_coeffs2)])

    results_ensemble_gaussian = np.array(results_ensemble_gaussian)
    results_ensemble_blackman = np.array(results_ensemble_blackman)
    print(np.average(results_ensemble_blackman, axis=0))
    print(np.average(results_ensemble_gaussian, axis=0))


def test_scale_normalize():
    spike_verif_path = f'./figures/test_normalized_scaled//spike_verif/'
    if not os.path.exists(spike_verif_path):
        os.makedirs(spike_verif_path)

    spikes, labels = ds.get_dataset_simulation(simNr=1, align_to_peak=False, normalize_spike=False, scale_spike=False)
    spikes_normalized, labels = ds.get_dataset_simulation(simNr=1, align_to_peak=False, normalize_spike=True, scale_spike=False)
    spikes_scaled, labels = ds.get_dataset_simulation(simNr=1, align_to_peak=False, normalize_spike=False, scale_spike=True)

    plt.plot(np.arange(len(spikes[0])), spikes[0], label="original")
    plt.plot(np.arange(len(spikes_normalized[0])), spikes_normalized[0], label="normalized")
    plt.plot(np.arange(len(spikes_scaled[0])), spikes_scaled[0], label="scaled")
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend(loc="upper left")
    plt.title(f"Verify spike {0}")
    plt.savefig(f'{spike_verif_path}spikes0')


def get_type(on_type, fft_real, fft_imag):
    if on_type == "real":
        spikes = fft_real
    elif on_type == "imag":
        spikes = fft_imag
    elif on_type == "magnitude":
        spikes = np.sqrt(fft_real * fft_real + fft_imag * fft_imag)
    elif on_type == "power":
        spikes = fft_real * fft_real + fft_imag * fft_imag
    elif on_type == "phase":
        spikes = np.arctan2(fft_imag, fft_real)
    elif on_type == "concatenated":
        power = fft_real * fft_real + fft_imag * fft_imag
        phase = np.arctan2(fft_imag, fft_real)
        spikes = np.concatenate((power, phase), axis=1)

    return spikes


def main_fft(program, case, alignment, on_type):
    range_min = 1
    range_max = 96
    epochs = 100
    spike_verif_path = f'./figures/fft/c{autoencoder_code_size}/{on_type}/{"wA" if alignment else "woA"}/{case}/spike_verif'
    plot_path = f'./figures/fft/c{autoencoder_code_size}/{on_type}/{"wA" if alignment else "woA"}/{case}/'
    weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft-{on_type}-{case}_{"wA" if alignment else "woA"}'

    if not os.path.exists(spike_verif_path):
        os.makedirs(spike_verif_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    fft_real, fft_imag = fft_input.apply_fft_on_range(case, alignment, range_min, 2)

    fft_real = np.array(fft_real)
    fft_imag = np.array(fft_imag)

    spikes = get_type(on_type, fft_real, fft_imag)

    spikes = np.array(spikes)

    autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                   encoder_layer_sizes=autoencoder_layer_sizes,
                                   decoder_layer_sizes=autoencoder_layer_sizes,
                                   code_size=autoencoder_code_size)

    encoder, autoenc = autoencoder.return_encoder()

    if program == "train":
        autoencoder.train(spikes, epochs=epochs)
        autoencoder.save_weights(weights_path)

        verify_output(spikes, encoder, autoenc, path=spike_verif_path)
        verify_random_outputs(spikes, encoder, autoenc, 10, path=spike_verif_path)

    if program == "test":
        autoencoder.load_weights(weights_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44:
                continue

            fft_real, fft_imag, labels = fft_input.apply_fft_on_sim(sim_nr=simulation_number, case=case,
                                                                    alignment=alignment)
            fft_real = np.array(fft_real)
            fft_imag = np.array(fft_imag)

            spikes = get_type(on_type, fft_real, fft_imag)

            spikes = spikes[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

            pn = 25
            labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

            scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                   marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
    if program == "validation":

        results_fft = []
        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44:
                continue

            fft_real, fft_imag, labels = fft_input.apply_fft_on_sim(sim_nr=simulation_number, case=case,
                                                                    alignment=alignment)
            fft_real = np.array(fft_real)
            fft_imag = np.array(fft_imag)

            spikes = get_type(on_type, fft_real, fft_imag)

            spikes = spikes[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)


            # scatter_plot.plot(f'GT{len(autoencoder_features)}', autoencoder_features, labels, marker='o')
            # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm')
            #
            # scatter_plot.plot(f'GT{len(pca_features)}', pca_features, labels, marker='o')
            # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca')


            pn = 25
            try:
                clustering_lstm_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                # scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, clustering_lstm_labels,
                #                        marker='o')
                # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm_sbm')
                #
                # scatter_plot.plot_grid('SBM' + str(len(pca_features)), pca_features, pn, clustering_pca_labels,
                #                        marker='o')
                # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca_sbm')

                hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_lstm_labels)

                scores_lstm = [adjusted_rand_score(labels, clustering_lstm_labels),
                          adjusted_mutual_info_score(labels, clustering_lstm_labels),
                          hom,
                          com,
                          vm,
                          calinski_harabasz_score(autoencoder_features, labels),
                          davies_bouldin_score(autoencoder_features, labels),
                          silhouette_score(autoencoder_features, labels)
                          ]

                results_fft.append(scores_lstm)

            except KeyError:
                pass

        results_fft = np.array(results_fft)

        mean_values = np.mean(results_fft, axis=0)

        print(f"FFT {on_type} - {alignment} - {case} -> ARI - {mean_values[0]}")
        print(f"FFT {on_type} - {alignment} - {case} -> AMI - {mean_values[1]}")
        print(f"FFT {on_type} - {alignment} - {case} -> Hom - {mean_values[2]}")
        print(f"FFT {on_type} - {alignment} - {case} -> Com - {mean_values[3]}")
        print(f"FFT {on_type} - {alignment} - {case} -> VM - {mean_values[4]}")
        print(f"FFT {on_type} - {alignment} - {case} -> CHS - {mean_values[5]}")
        print(f"FFT {on_type} - {alignment} - {case} -> DBS - {mean_values[6]}")
        print(f"FFT {on_type} - {alignment} - {case} -> SS - {mean_values[7]}")


def main_ensemble_algebraic(program, algebraic_combination):
    range_min = 1
    range_max = 96
    epochs = 100
    alignment = 2 # align to Na+ peak
    on_type = "magnitude"
    gaussian_weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft_windowed--gaussian-{on_type}_align{alignment}'
    blackman_weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft_windowed--blackman-{on_type}_align{alignment}'
    plot_path = f'./figures/fft_windowed_ensemble_{algebraic_combination}/'

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    fft_real_blackman, fft_imag_blackman, labels = fft_input.apply_fft_windowed_on_range(alignment, range_min, range_max, "blackman")
    fft_real_gaussian, fft_imag_gaussian, labels = fft_input.apply_fft_windowed_on_range(alignment, range_min, range_max, "gaussian")

    fft_real_blackman = np.array(fft_real_blackman)
    fft_imag_blackman = np.array(fft_imag_blackman)
    fft_real_gaussian = np.array(fft_real_gaussian)
    fft_imag_gaussian = np.array(fft_imag_gaussian)

    spikes_blackman = get_type(on_type, fft_real_blackman, fft_imag_blackman)
    spikes_blackman = np.array(spikes_blackman)
    spikes_gaussian = get_type(on_type, fft_real_gaussian, fft_imag_gaussian)
    spikes_gaussian = np.array(spikes_gaussian)

    model_list = []

    gaussian_autoencoder = AutoencoderModel(input_size=len(spikes_gaussian[0]),
                                   encoder_layer_sizes=autoencoder_layer_sizes,
                                   decoder_layer_sizes=autoencoder_layer_sizes,
                                   code_size=autoencoder_code_size)

    blackman_autoencoder = AutoencoderModel(input_size=len(spikes_blackman[0]),
                                   encoder_layer_sizes=autoencoder_layer_sizes,
                                   decoder_layer_sizes=autoencoder_layer_sizes,
                                   code_size=autoencoder_code_size)

    model_list.append(gaussian_autoencoder)
    model_list.append(blackman_autoencoder)

    gaussian_encoder, gaussian_autoenc = gaussian_autoencoder.return_encoder()
    blackman_encoder, blackman_autoenc = blackman_autoencoder.return_encoder()


    if program == "train":
        gaussian_autoencoder.train(spikes_gaussian, epochs=epochs)
        # gaussian_autoencoder.save_weights(weights_path)

        blackman_autoencoder.train(spikes_blackman, epochs=epochs)
        # blackman_autoencoder.save_weights(weights_path)
    elif program == "test":
        blackman_autoencoder.load_weights(blackman_weights_path)
        gaussian_autoencoder.load_weights(gaussian_weights_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number== 78:
                continue

            fft_real_blackman, fft_imag_blackman, labels_blackman = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                                                        alignment=alignment,
                                                                                                        window_type="blackman")
            fft_real_gaussian, fft_imag_gaussian, labels_gaussian = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                                                        alignment=alignment,
                                                                                                        window_type="gaussian")

            fft_real_blackman = np.array(fft_real_blackman)
            fft_imag_blackman = np.array(fft_imag_blackman)
            fft_real_gaussian = np.array(fft_real_gaussian)
            fft_imag_gaussian = np.array(fft_imag_gaussian)

            spikes_blackman = get_type(on_type, fft_real_blackman, fft_imag_blackman)
            spikes_blackman = np.array(spikes_blackman)
            spikes_gaussian = get_type(on_type, fft_real_gaussian, fft_imag_gaussian)
            spikes_gaussian = np.array(spikes_gaussian)

            spikes_gaussian = spikes_gaussian[labels_gaussian != 0]
            labels_gaussian = labels_gaussian[labels_gaussian != 0]
            spikes_blackman = spikes_blackman[labels_blackman != 0]
            labels_blackman = labels_blackman[labels_blackman != 0]

            autoencoder_features_blackman = get_codes(spikes_blackman, gaussian_encoder)
            autoencoder_features_gaussian = get_codes(spikes_gaussian, blackman_encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features_blackman = pca_2d.fit_transform(autoencoder_features_blackman)
            autoencoder_features_gaussian = pca_2d.fit_transform(autoencoder_features_gaussian)


            if algebraic_combination == "mean":
                autoencoder_features = (autoencoder_features_blackman + autoencoder_features_gaussian) / 2
            elif algebraic_combination == "sum":
                autoencoder_features = autoencoder_features_blackman + autoencoder_features_gaussian
            elif algebraic_combination == "product":
                autoencoder_features = autoencoder_features_blackman * autoencoder_features_gaussian
            elif algebraic_combination == "min":
                autoencoder_features = np.minimum(autoencoder_features_blackman, autoencoder_features_gaussian)
            elif algebraic_combination == "max":
                autoencoder_features = np.maximum(autoencoder_features_blackman, autoencoder_features_gaussian)

            scatter_plot.plot('GT' + str(len(autoencoder_features_blackman)), autoencoder_features_blackman, labels_gaussian, marker='o')
            plt.savefig(plot_path + f'gt_sim{simulation_number}_gaussian')
            scatter_plot.plot('GT' + str(len(autoencoder_features_gaussian)), autoencoder_features_gaussian, labels_blackman, marker='o')
            plt.savefig(plot_path + f'gt_sim{simulation_number}_blackman')

            # from sklearn.ensemble import VotingClassifier
            # from sklearn import model_selection
            # kfold = model_selection.KFold(n_splits=10, random_state=7)
            # # create the sub models
            # estimators = []
            # estimators.append(('gaussian', gaussian_encoder))
            # estimators.append(('blackman', blackman_encoder))
            # # create the ensemble model
            # ensemble = VotingClassifier(estimators)
            # results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
            # print(results.mean())
    elif program == "validate":
        blackman_autoencoder.load_weights(blackman_weights_path)
        gaussian_autoencoder.load_weights(gaussian_weights_path)

        results_ensemble = []
        results_blackman = []
        results_gaussian = []
        print(f"-------------{algebraic_combination}-------------")
        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                continue

            fft_real_blackman, fft_imag_blackman, labels_blackman = fft_input.apply_fft_windowed_on_sim(
                sim_nr=simulation_number,
                alignment=alignment,
                window_type="blackman")
            fft_real_gaussian, fft_imag_gaussian, labels_gaussian = fft_input.apply_fft_windowed_on_sim(
                sim_nr=simulation_number,
                alignment=alignment,
                window_type="gaussian")

            fft_real_blackman = np.array(fft_real_blackman)
            fft_imag_blackman = np.array(fft_imag_blackman)
            fft_real_gaussian = np.array(fft_real_gaussian)
            fft_imag_gaussian = np.array(fft_imag_gaussian)

            spikes_blackman = get_type(on_type, fft_real_blackman, fft_imag_blackman)
            spikes_blackman = np.array(spikes_blackman)
            spikes_gaussian = get_type(on_type, fft_real_gaussian, fft_imag_gaussian)
            spikes_gaussian = np.array(spikes_gaussian)

            spikes_gaussian = spikes_gaussian[labels_gaussian != 0]
            labels_gaussian = labels_gaussian[labels_gaussian != 0]
            spikes_blackman = spikes_blackman[labels_blackman != 0]
            labels_blackman = labels_blackman[labels_blackman != 0]

            autoencoder_features_blackman = get_codes(spikes_blackman, gaussian_encoder)
            autoencoder_features_gaussian = get_codes(spikes_gaussian, blackman_encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features_blackman = pca_2d.fit_transform(autoencoder_features_blackman)
            autoencoder_features_gaussian = pca_2d.fit_transform(autoencoder_features_gaussian)

            autoencoder_features = None
            if algebraic_combination == "mean":
                autoencoder_features = (autoencoder_features_blackman + autoencoder_features_gaussian) / 2
            elif algebraic_combination == "sum":
                autoencoder_features = autoencoder_features_blackman + autoencoder_features_gaussian
            elif algebraic_combination == "prod":
                autoencoder_features = autoencoder_features_blackman * autoencoder_features_gaussian
            elif algebraic_combination == "min":
                autoencoder_features = np.minimum(autoencoder_features_blackman, autoencoder_features_gaussian)
            elif algebraic_combination == "max":
                autoencoder_features = np.maximum(autoencoder_features_blackman, autoencoder_features_gaussian)

            clustering_labels = SBM.best(autoencoder_features, 25, ccThreshold=5, version=2)
            sil_coeffs = silhouette_samples(autoencoder_features, labels_blackman, metric='mahalanobis')

            results_ensemble.append([adjusted_rand_score(labels_blackman, clustering_labels),
                                    adjusted_mutual_info_score(labels_blackman, clustering_labels),
                                     np.average(sil_coeffs)])
            clustering_labels = SBM.best(autoencoder_features_blackman, 25, ccThreshold=5, version=2)
            sil_coeffs = silhouette_samples(autoencoder_features_blackman, labels_blackman, metric='mahalanobis')
            results_blackman.append([adjusted_rand_score(labels_blackman, clustering_labels),
                            adjusted_mutual_info_score(labels_blackman, clustering_labels),
                                     np.average(sil_coeffs)])
            clustering_labels = SBM.best(autoencoder_features_gaussian, 25, ccThreshold=5, version=2)
            sil_coeffs = silhouette_samples(autoencoder_features_gaussian, labels_gaussian, metric='mahalanobis')
            results_gaussian.append([adjusted_rand_score(labels_gaussian, clustering_labels),
                            adjusted_mutual_info_score(labels_gaussian, clustering_labels),
                                     np.average(sil_coeffs)])

        results_ensemble = np.array(results_ensemble)
        results_blackman = np.array(results_blackman)
        results_gaussian = np.array(results_gaussian)
        print(np.average(results_ensemble, axis=0))
        print(np.average(results_blackman, axis=0))
        print(np.average(results_gaussian, axis=0))


def main_fft_windowed(program, alignment, on_type, window_type):
    range_min = 1
    range_max = 96
    epochs = 100
    spike_verif_path = f'./figures/fft_windowed/c{autoencoder_code_size}/{window_type}/{on_type}/align{alignment}/spike_verif'
    plot_path = f'./figures/fft_windowed/c{autoencoder_code_size}/{window_type}/{on_type}/align{alignment}/'
    weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft_windowed--{window_type}-{on_type}_align{alignment}'

    if not os.path.exists(spike_verif_path):
        os.makedirs(spike_verif_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_range(alignment, range_min, 2, window_type)

    fft_real = np.array(fft_real)
    fft_imag = np.array(fft_imag)

    spikes = get_type(on_type, fft_real, fft_imag)
    spikes = np.array(spikes)

    autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                   encoder_layer_sizes=autoencoder_layer_sizes,
                                   decoder_layer_sizes=autoencoder_layer_sizes,
                                   code_size=autoencoder_code_size)

    encoder, autoenc = autoencoder.return_encoder()

    if program == "train":
        autoencoder.train(spikes, epochs=epochs)
        autoencoder.save_weights(weights_path)

        verify_output(spikes, encoder, autoenc, path=spike_verif_path)
        verify_random_outputs(spikes, encoder, autoenc, 10, path=spike_verif_path)

    if program == "test":
        autoencoder.load_weights(weights_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number== 78:
                continue

            fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                             alignment=alignment,
                                                                             window_type=window_type)
            fft_real = np.array(fft_real)
            fft_imag = np.array(fft_imag)

            spikes = get_type(on_type, fft_real, fft_imag)
            spikes = np.array(spikes)

            spikes = spikes[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

            pn = 25
            labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
            try:
                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
            except KeyError:
                pass
    if program == "validation":
        results_fft = []
        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number== 78:
                continue

            fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                             alignment=alignment,
                                                                             window_type=window_type)
            fft_real = np.array(fft_real)
            fft_imag = np.array(fft_imag)

            spikes = get_type(on_type, fft_real, fft_imag)
            spikes = np.array(spikes)

            spikes = spikes[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)


            # scatter_plot.plot(f'GT{len(autoencoder_features)}', autoencoder_features, labels, marker='o')
            # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm')
            #
            # scatter_plot.plot(f'GT{len(pca_features)}', pca_features, labels, marker='o')
            # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca')


            pn = 25
            try:
                clustering_lstm_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                # scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, clustering_lstm_labels,
                #                        marker='o')
                # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm_sbm')
                #
                # scatter_plot.plot_grid('SBM' + str(len(pca_features)), pca_features, pn, clustering_pca_labels,
                #                        marker='o')
                # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca_sbm')

                hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_lstm_labels)

                scores_lstm = [adjusted_rand_score(labels, clustering_lstm_labels),
                          adjusted_mutual_info_score(labels, clustering_lstm_labels),
                          hom,
                          com,
                          vm,
                          calinski_harabasz_score(autoencoder_features, labels),
                          davies_bouldin_score(autoencoder_features, labels),
                          silhouette_score(autoencoder_features, labels)
                          ]

                results_fft.append(scores_lstm)

            except KeyError:
                pass

        results_fft = np.array(results_fft)

        mean_values = np.mean(results_fft, axis=0)

        print(f"FFT {window_type} - {on_type} - {alignment} -> ARI - {mean_values[0]}")
        print(f"FFT {window_type} - {on_type} - {alignment} -> AMI - {mean_values[1]}")
        print(f"FFT {window_type} - {on_type} - {alignment} -> Hom - {mean_values[2]}")
        print(f"FFT {window_type} - {on_type} - {alignment} -> Com - {mean_values[3]}")
        print(f"FFT {window_type} - {on_type} - {alignment} -> VM - {mean_values[4]}")
        print(f"FFT {window_type} - {on_type} - {alignment} -> CHS - {mean_values[5]}")
        print(f"FFT {window_type} - {on_type} - {alignment} -> DBS - {mean_values[6]}")
        print(f"FFT {window_type} - {on_type} - {alignment} -> SS - {mean_values[7]}")


def test_fft_windowed(window_type):
    def verify_output(spikes, windowed_spikes, i=0, path=""):
        plt.plot(np.arange(len(spikes[i])), spikes[i])
        plt.plot(np.arange(len(windowed_spikes[i])), windowed_spikes[i])
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.title(f"Verify spike {i}")
        plt.savefig(f'{path}/spike{i}')
        plt.clf()
        # plt.show()

    def verify_random_outputs(spikes, windowed_spikes, verifications=0, path=""):
        random_list = np.random.choice(range(len(spikes)), verifications, replace=False)

        for random_index in random_list:
            verify_output(spikes, windowed_spikes, random_index, path)

    for alignment in [True, False, 2, 3]:
        plot_path = f'./figures/fft/blackman_test/align{alignment}'

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        spikes, labels = ds.get_dataset_simulation(simNr=1, align_to_peak=alignment)

        if window_type == "blackman":
            windowed_spikes = fft_input.apply_blackman_window(spikes)
        # std value from 5-15 to get 68% in mean-std:mean+std
        elif window_type == "gaussian":
            windowed_spikes = fft_input.apply_gaussian_window(spikes, std=10)

        # verify_random_outputs(spikes, windowed_spikes, 10, plot_path)
        verify_output(spikes, windowed_spikes, 685, path=plot_path)
        verify_output(spikes, windowed_spikes, 2171, path=plot_path)
        verify_output(spikes, windowed_spikes, 2592, path=plot_path)


def main_dpss(program):
    # M = 79
    # NW = 4
    # win, eigvals = windows.dpss(M, NW, 5, return_ratios=True)

    # print(win)
    #
    # fig, ax = plt.subplots(1)
    # ax.plot(win.T, linewidth=1.)
    # ax.set(xlim=[0, M - 1], ylim=[-0.3, 0.3], xlabel='Samples',
    #        title='DPSS, M=%d, NW=%0.1f' % (M, NW))
    # ax.legend(['win[%d] (%0.4f)' % (ii, ratio)
    #            for ii, ratio in enumerate(eigvals)])
    # fig.tight_layout()
    # plt.show()

    range_min = 1
    range_max = 96
    epochs = 100
    align = 2
    window_type = "dpss"
    on_type = "magnitude"
    spike_verif_path = f'./figures/fft_dpss/spike_verif'
    plot_path = f'./figures/fft_dpss/'
    weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft_dpss'

    if not os.path.exists(spike_verif_path):
        os.makedirs(spike_verif_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_range(alignment=align, range_min=range_min, range_max=2, window_type=window_type)

    fft_real = np.array(fft_real)
    fft_imag = np.array(fft_imag)

    spikes = get_type(on_type, fft_real, fft_imag)
    spikes = np.array(spikes)

    autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                   encoder_layer_sizes=autoencoder_layer_sizes,
                                   decoder_layer_sizes=autoencoder_layer_sizes,
                                   code_size=autoencoder_code_size)

    encoder, autoenc = autoencoder.return_encoder()

    if program == "train":
        autoencoder.train(spikes, epochs=epochs)
        autoencoder.save_weights(weights_path)

        verify_output(spikes, encoder, autoenc, i=0, path=spike_verif_path)
        verify_output(spikes, encoder, autoenc, i=1, path=spike_verif_path)
        verify_output(spikes, encoder, autoenc, i=2, path=spike_verif_path)
        verify_output(spikes, encoder, autoenc, i=3, path=spike_verif_path)
        verify_output(spikes, encoder, autoenc, i=4, path=spike_verif_path)
        verify_random_outputs(spikes, encoder, autoenc, 10, path=spike_verif_path)

    if program == "test":
        autoencoder.load_weights(weights_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number== 78:
                continue

            fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                             alignment=align,
                                                                             window_type="none")
            fft_real = np.array(fft_real)
            fft_imag = np.array(fft_imag)

            spikes = get_type(on_type, fft_real, fft_imag)
            spikes = np.array(spikes)

            spikes = spikes[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

            pn = 25
            labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
            try:
                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
            except KeyError:
                pass

    if program == "validation":
        results_fft = []
        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44:
                continue

            fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                             alignment=align,
                                                                             window_type="none")
            fft_real = np.array(fft_real)
            fft_imag = np.array(fft_imag)

            spikes = get_type(on_type, fft_real, fft_imag)
            spikes = np.array(spikes)

            spikes = spikes[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)


            # scatter_plot.plot(f'GT{len(autoencoder_features)}', autoencoder_features, labels, marker='o')
            # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm')
            #
            # scatter_plot.plot(f'GT{len(pca_features)}', pca_features, labels, marker='o')
            # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca')


            pn = 25
            try:
                clustering_lstm_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                # scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, clustering_lstm_labels,
                #                        marker='o')
                # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_lstm_sbm')
                #
                # scatter_plot.plot_grid('SBM' + str(len(pca_features)), pca_features, pn, clustering_pca_labels,
                #                        marker='o')
                # plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca_sbm')

                hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_lstm_labels)

                scores_lstm = [adjusted_rand_score(labels, clustering_lstm_labels),
                          adjusted_mutual_info_score(labels, clustering_lstm_labels),
                          hom,
                          com,
                          vm,
                          calinski_harabasz_score(autoencoder_features, labels),
                          davies_bouldin_score(autoencoder_features, labels),
                          silhouette_score(autoencoder_features, labels)
                          ]

                results_fft.append(scores_lstm)

            except KeyError:
                pass

        results_fft = np.array(results_fft)

        mean_values = np.mean(results_fft, axis=0)

        print(f"FFT {on_type} -> ARI - {mean_values[0]}")
        print(f"FFT {on_type} -> AMI - {mean_values[1]}")
        print(f"FFT {on_type} -> Hom - {mean_values[2]}")
        print(f"FFT {on_type} -> Com - {mean_values[3]}")
        print(f"FFT {on_type} -> VM - {mean_values[4]}")
        print(f"FFT {on_type} -> CHS - {mean_values[5]}")
        print(f"FFT {on_type} -> DBS - {mean_values[6]}")
        print(f"FFT {on_type} -> SS - {mean_values[7]}")


def main_iterative_dpss(program):
    # M = 79
    # NW = 4
    # win, eigvals = windows.dpss(M, NW, 5, return_ratios=True)

    # print(win)
    #
    # fig, ax = plt.subplots(1)
    # ax.plot(win.T, linewidth=1.)
    # ax.set(xlim=[0, M - 1], ylim=[-0.3, 0.3], xlabel='Samples',
    #        title='DPSS, M=%d, NW=%0.1f' % (M, NW))
    # ax.legend(['win[%d] (%0.4f)' % (ii, ratio)
    #            for ii, ratio in enumerate(eigvals)])
    # fig.tight_layout()
    # plt.show()

    range_min = 1
    range_max = 96
    epochs = 100
    align = 2
    window_type = "dpss"
    on_type = "magnitude"
    spike_verif_path = f'./figures/fft_dpss_c{autoencoder_code_size}/spike_verif'
    plot_path = f'./figures/fft_dpss_c{autoencoder_code_size}/'
    weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft_dpss'

    if not os.path.exists(spike_verif_path):
        os.makedirs(spike_verif_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_range(alignment=align, range_min=range_min, range_max=2, window_type=window_type)

    fft_real = np.array(fft_real)
    fft_imag = np.array(fft_imag)

    spikes = get_type(on_type, fft_real, fft_imag)
    spikes = np.array(spikes)

    autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                   encoder_layer_sizes=autoencoder_layer_sizes,
                                   decoder_layer_sizes=autoencoder_layer_sizes,
                                   code_size=autoencoder_code_size)

    encoder, autoenc = autoencoder.return_encoder()

    if program == "train":
        autoencoder.train(spikes, epochs=epochs)
        autoencoder.save_weights(weights_path)

        verify_output(spikes, encoder, autoenc, i=0, path=spike_verif_path)
        verify_output(spikes, encoder, autoenc, i=1, path=spike_verif_path)
        verify_output(spikes, encoder, autoenc, i=2, path=spike_verif_path)
        verify_output(spikes, encoder, autoenc, i=3, path=spike_verif_path)
        verify_output(spikes, encoder, autoenc, i=4, path=spike_verif_path)
        verify_random_outputs(spikes, encoder, autoenc, 10, path=spike_verif_path)

    if program == "test":
        autoencoder.load_weights(weights_path)

        iterative_spikes = []
        iterative_labels = []

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number== 78:
                continue

            fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                             alignment=align,
                                                                             window_type="none")
            fft_real = np.array(fft_real)
            fft_imag = np.array(fft_imag)

            spikes = get_type(on_type, fft_real, fft_imag)
            spikes = np.array(spikes)

            spikes = spikes[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

            pn = 25
            sbm_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
            try:
                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, sbm_labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
            except KeyError:
                pass

            kmeans = KMeans(n_clusters=2, random_state=0).fit(autoencoder_features)
            kmeans_labels = kmeans.labels_
            scatter_plot.plot('SBM' + str(len(autoencoder_features)), autoencoder_features, kmeans_labels,
                                   marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_kmeans')


            first_cluster = autoencoder_features[kmeans_labels == 0]
            second_cluster = autoencoder_features[kmeans_labels == 1]
            min_distance = 10000000
            for first_point in first_cluster:
                for second_point in second_cluster:
                    if min_distance > distance(first_point, second_point):
                        min_distance = distance(first_point, second_point)

            if min_distance > 0.25:
                if np.mean(first_cluster, axis=0)[0] < np.mean(second_cluster, axis=0)[0]:
                    unique = np.unique(labels[kmeans_labels == 0])
                    print(simulation_number, len(unique), unique)
                    if len(unique) > 1:
                        iterative_spikes.append(spikes[kmeans_labels == 0])
                        iterative_labels.append(labels[kmeans_labels == 0])
                else:
                    unique = np.unique(labels[kmeans_labels == 1])
                    print(simulation_number, len(unique), unique)
                    if len(unique) > 1:
                        iterative_spikes.append(spikes[kmeans_labels == 1])
                        iterative_labels.append(labels[kmeans_labels == 1])

        new_spikes = np.vstack(iterative_spikes)
        autoencoder = AutoencoderModel(input_size=len(new_spikes[0]),
                                       encoder_layer_sizes=autoencoder_layer_sizes,
                                       decoder_layer_sizes=autoencoder_layer_sizes,
                                       code_size=autoencoder_code_size)

        spike_verif_path = f'./figures/fft_dpss_iter2_c{autoencoder_code_size}/spike_verif'
        plot_path = f'./figures/fft_dpss_iter2_c{autoencoder_code_size}/'
        weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft_dpss_iter2'

        if not os.path.exists(spike_verif_path):
            os.makedirs(spike_verif_path)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        autoencoder.train(new_spikes, epochs=epochs)
        autoencoder.save_weights(weights_path)
        autoencoder.load_weights(weights_path)

        simulation_number = 0
        for i in range(len(iterative_spikes)):
            autoencoder_features = get_codes(iterative_spikes[i], encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, iterative_labels[i], marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

            pn = 25
            labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
            try:
                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
            except KeyError:
                pass

            kmeans = KMeans(n_clusters=2, random_state=0).fit(autoencoder_features)
            kmeans_labels = kmeans.labels_
            scatter_plot.plot('SBM' + str(len(autoencoder_features)), autoencoder_features, kmeans_labels,
                              marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_kmeans')

            simulation_number+=1


def main_dpss_test(program):
    # M = 79
    # NW = 4
    # win, eigvals = windows.dpss(M, NW, 5, return_ratios=True)

    # print(win)
    #
    # fig, ax = plt.subplots(1)
    # ax.plot(win.T, linewidth=1.)
    # ax.set(xlim=[0, M - 1], ylim=[-0.3, 0.3], xlabel='Samples',
    #        title='DPSS, M=%d, NW=%0.1f' % (M, NW))
    # ax.legend(['win[%d] (%0.4f)' % (ii, ratio)
    #            for ii, ratio in enumerate(eigvals)])
    # fig.tight_layout()
    # plt.show()

    range_min = 1
    range_max = 2
    epochs = 100
    align = 2
    window_type = "dpss"
    on_type = "magnitude"
    spike_verif_path = f'./figures/fft_dpss_test/spike_verif'
    plot_path = f'./figures/fft_dpss_test/'
    weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft_dpss'

    if not os.path.exists(spike_verif_path):
        os.makedirs(spike_verif_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_range(alignment=align, range_min=range_min, range_max=range_max, window_type=window_type)

    fft_real = np.array(fft_real)
    fft_imag = np.array(fft_imag)

    spikes = get_type(on_type, fft_real, fft_imag)
    spikes = np.array(spikes)

    autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                   encoder_layer_sizes=autoencoder_layer_sizes,
                                   decoder_layer_sizes=autoencoder_layer_sizes,
                                   code_size=autoencoder_code_size)

    encoder, autoenc = autoencoder.return_encoder()

    if program == "sim4":
        autoencoder.load_weights(weights_path)

        simulation_number = 4
        fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                         alignment=align,
                                                                         window_type="none")
        fft_real = np.array(fft_real)
        fft_imag = np.array(fft_imag)

        spikes = get_type(on_type, fft_real, fft_imag)
        spikes = np.array(spikes)

        spikes = spikes[labels != 0]
        labels = labels[labels != 0]

        spikes = spikes[labels != 4]
        labels = labels[labels != 4]

        autoencoder_features = get_codes(spikes, encoder)

        pca_2d = PCA(n_components=2)
        autoencoder_features = pca_2d.fit_transform(autoencoder_features)

        scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
        plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

        pn = 25
        labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
        try:
            scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                   marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
        except KeyError:
            pass

    if program == "sim13":
        autoencoder.load_weights(weights_path)

        simulation_number = 13
        fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                         alignment=align,
                                                                         window_type="none")
        fft_real = np.array(fft_real)
        fft_imag = np.array(fft_imag)

        spikes = get_type(on_type, fft_real, fft_imag)
        spikes = np.array(spikes)

        spikes = spikes[labels != 0]
        labels = labels[labels != 0]

        spikes = spikes[labels != 5]
        labels = labels[labels != 5]

        spikes = spikes[labels != 6]
        labels = labels[labels != 6]

        autoencoder_features = get_codes(spikes, encoder)

        pca_2d = PCA(n_components=2)
        autoencoder_features = pca_2d.fit_transform(autoencoder_features)

        scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
        plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

        pn = 25
        labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
        try:
            scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                   marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
        except KeyError:
            pass

    if program == "sim26":
        autoencoder.load_weights(weights_path)

        simulation_number = 26
        fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                         alignment=align,
                                                                         window_type="none")
        fft_real = np.array(fft_real)
        fft_imag = np.array(fft_imag)

        spikes = get_type(on_type, fft_real, fft_imag)
        spikes = np.array(spikes)

        spikes = spikes[labels != 0]
        labels = labels[labels != 0]

        spikes = spikes[labels != 2]
        labels = labels[labels != 2]

        spikes = spikes[labels != 3]
        labels = labels[labels != 3]

        autoencoder_features = get_codes(spikes, encoder)

        pca_2d = PCA(n_components=2)
        autoencoder_features = pca_2d.fit_transform(autoencoder_features)

        scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
        plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

        pn = 25
        labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
        try:
            scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                   marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
        except KeyError:
            pass

    if program == "sim40":
        autoencoder.load_weights(weights_path)

        simulation_number = 40
        fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                         alignment=align,
                                                                         window_type="none")
        fft_real = np.array(fft_real)
        fft_imag = np.array(fft_imag)

        spikes = get_type(on_type, fft_real, fft_imag)
        spikes = np.array(spikes)

        spikes = spikes[labels != 0]
        labels = labels[labels != 0]

        spikes = spikes[labels != 2]
        labels = labels[labels != 2]

        spikes = spikes[labels != 7]
        labels = labels[labels != 7]

        autoencoder_features = get_codes(spikes, encoder)

        pca_2d = PCA(n_components=2)
        autoencoder_features = pca_2d.fit_transform(autoencoder_features)

        scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
        plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

        pn = 25
        labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
        try:
            scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                   marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
        except KeyError:
            pass


def main_dpss_cascade():
    range_min = 1
    range_max = 96
    epochs = 100
    align = 2
    window_type = "dpss"
    on_type = "magnitude"

    spikes, labels = dsa.stack_simulations_range(range_min, range_max, True, True, alignment=align)

    M = 79
    NW = 4
    nr_windows = 5
    win, eigvals = windows.dpss(M, NW, nr_windows, return_ratios=True)


    code_list = []

    all_window_codes = []
    for i in range(len(win)):
        stacked_code_list = []
        window_codes = []

        spike_verif_path = f'./figures/fft_dpss_cascade/spike_verif{i}'
        plot_path = f'./figures/fft_dpss_cascade/window{i}/'
        weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft_dpss_cascade{i}'

        if not os.path.exists(spike_verif_path):
            os.makedirs(spike_verif_path)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        windowed_spikes = spikes * win[i]

        # PADDED SPIKE
        fft_real, fft_imag = fft_padded_spike(windowed_spikes)

        fft_real = np.array(fft_real)
        fft_imag = np.array(fft_imag)


        spikes_fft = get_type(on_type, fft_real, fft_imag)
        spikes_fft = np.array(spikes_fft)

        autoencoder = AutoencoderModel(input_size=len(spikes_fft[0]),
                                       encoder_layer_sizes=autoencoder_layer_sizes,
                                       decoder_layer_sizes=autoencoder_layer_sizes,
                                       code_size=autoencoder_code_size)

        encoder, autoenc = autoencoder.return_encoder()


        autoencoder.train(spikes_fft, epochs=epochs)
        autoencoder.save_weights(weights_path)

        verify_output(spikes_fft, encoder, autoenc, i=0, path=spike_verif_path)
        verify_random_outputs(spikes_fft, encoder, autoenc, 10, path=spike_verif_path)


        autoencoder.load_weights(weights_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number== 78:
                continue

            fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                             alignment=align,
                                                                             window_type="none")
            fft_real = np.array(fft_real)
            fft_imag = np.array(fft_imag)

            spikes_fft = get_type(on_type, fft_real, fft_imag)
            spikes_fft = np.array(spikes_fft)

            spikes_fft = spikes_fft[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes_fft, encoder)

            stacked_code_list.append(autoencoder_features)
            window_codes.append(autoencoder_features)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

            pn = 25
            labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
            try:
                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
            except KeyError:
                pass

        code_list.append(np.vstack(stacked_code_list))
        all_window_codes.append(window_codes)


    stacked_codes = np.hstack(code_list)

    spike_verif_path = f'./figures/fft_dpss_cascade/spike_verif'
    plot_path = f'./figures/fft_dpss_cascade/'
    weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft_dpss_cascade'

    if not os.path.exists(spike_verif_path):
        os.makedirs(spike_verif_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    autoencoder = AutoencoderModel(input_size=len(stacked_codes[0]),
                                   encoder_layer_sizes=autoencoder_cascade_layer_sizes,
                                   decoder_layer_sizes=autoencoder_cascade_layer_sizes,
                                   code_size=autoencoder_code_size)

    encoder, autoenc = autoencoder.return_encoder()

    autoencoder.train(stacked_codes, epochs=epochs)
    autoencoder.save_weights(weights_path)

    verify_output(stacked_codes, encoder, autoenc, i=0, path=spike_verif_path)
    verify_random_outputs(stacked_codes, encoder, autoenc, 10, path=spike_verif_path)

    autoencoder.load_weights(weights_path)

    index = range_min
    for simulation_number in range(range_min, range_max):
        if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
            continue

        _, _, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                           alignment=align,
                                                           window_type="none")
        labels = labels[labels != 0]

        sim_window_codes = []
        for i in range(len(all_window_codes)):
            sim_window_codes.append(all_window_codes[i][index-1])

        index += 1

        sim_codes = np.hstack(sim_window_codes)

        autoencoder_features = get_codes(sim_codes, encoder)

        pca_2d = PCA(n_components=2)
        autoencoder_features = pca_2d.fit_transform(autoencoder_features)

        scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
        plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

        pn = 25
        labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
        try:
            scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                   marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
        except KeyError:
            pass


def main_dpss_cascade_validation():
    range_min = 1
    range_max = 96
    epochs = 100
    align = 2
    window_type = "dpss"
    on_type = "magnitude"

    spikes, labels = dsa.stack_simulations_range(range_min, 2, True, True, alignment=align)

    M = 79
    NW = 4
    nr_windows = 5
    win, eigvals = windows.dpss(M, NW, nr_windows, return_ratios=True)


    code_list = []

    all_window_codes = []
    for i in range(len(win)):
        stacked_code_list = []
        window_codes = []

        weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft_dpss_cascade{i}'

        windowed_spikes = spikes * win[i]

        # PADDED SPIKE
        fft_real, fft_imag = fft_padded_spike(windowed_spikes)

        fft_real = np.array(fft_real)
        fft_imag = np.array(fft_imag)


        spikes_fft = get_type(on_type, fft_real, fft_imag)
        spikes_fft = np.array(spikes_fft)

        autoencoder = AutoencoderModel(input_size=len(spikes_fft[0]),
                                       encoder_layer_sizes=autoencoder_layer_sizes,
                                       decoder_layer_sizes=autoencoder_layer_sizes,
                                       code_size=autoencoder_code_size)

        encoder, autoenc = autoencoder.return_encoder()

        autoencoder.load_weights(weights_path)

        results_fft = []
        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number== 78:
                continue

            fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                             alignment=align,
                                                                             window_type="none")
            fft_real = np.array(fft_real)
            fft_imag = np.array(fft_imag)

            spikes_fft = get_type(on_type, fft_real, fft_imag)
            spikes_fft = np.array(spikes_fft)

            spikes_fft = spikes_fft[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes_fft, encoder)

            stacked_code_list.append(autoencoder_features)
            window_codes.append(autoencoder_features)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            pn = 25
            clustering_lstm_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

            hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_lstm_labels)

            scores_lstm = [adjusted_rand_score(labels, clustering_lstm_labels),
                           adjusted_mutual_info_score(labels, clustering_lstm_labels),
                           hom,
                           com,
                           vm,
                           calinski_harabasz_score(autoencoder_features, labels),
                           davies_bouldin_score(autoencoder_features, labels),
                           silhouette_score(autoencoder_features, labels)
                           ]

            results_fft.append(scores_lstm)

        code_list.append(np.vstack(stacked_code_list))
        all_window_codes.append(window_codes)

        results_fft = np.array(results_fft)

        mean_values = np.mean(results_fft, axis=0)

        print(f"FFT W{i} - {on_type}  -> ARI - {mean_values[0]}")
        print(f"FFT W{i} - {on_type}  -> AMI - {mean_values[1]}")
        print(f"FFT W{i} - {on_type}  -> Hom - {mean_values[2]}")
        print(f"FFT W{i} - {on_type}  -> Com - {mean_values[3]}")
        print(f"FFT W{i} - {on_type}  -> VM - {mean_values[4]}")
        print(f"FFT W{i} - {on_type}  -> CHS - {mean_values[5]}")
        print(f"FFT W{i} - {on_type}  -> DBS - {mean_values[6]}")
        print(f"FFT W{i} - {on_type}  -> SS - {mean_values[7]}")

    stacked_codes = np.hstack(code_list)

    weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft_dpss_cascade'

    autoencoder = AutoencoderModel(input_size=len(stacked_codes[0]),
                                   encoder_layer_sizes=autoencoder_cascade_layer_sizes,
                                   decoder_layer_sizes=autoencoder_cascade_layer_sizes,
                                   code_size=autoencoder_code_size)

    encoder, autoenc = autoencoder.return_encoder()

    autoencoder.load_weights(weights_path)

    index = range_min
    results_fft = []
    for simulation_number in range(range_min, range_max):
        if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
            continue

        _, _, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                           alignment=align,
                                                           window_type="none")
        labels = labels[labels != 0]

        sim_window_codes = []
        for i in range(len(all_window_codes)):
            sim_window_codes.append(all_window_codes[i][index-1])

        index += 1

        sim_codes = np.hstack(sim_window_codes)

        autoencoder_features = get_codes(sim_codes, encoder)

        pca_2d = PCA(n_components=2)
        autoencoder_features = pca_2d.fit_transform(autoencoder_features)

        pn = 25
        clustering_lstm_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

        hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_lstm_labels)

        scores_lstm = [adjusted_rand_score(labels, clustering_lstm_labels),
                       adjusted_mutual_info_score(labels, clustering_lstm_labels),
                       hom,
                       com,
                       vm,
                       calinski_harabasz_score(autoencoder_features, labels),
                       davies_bouldin_score(autoencoder_features, labels),
                       silhouette_score(autoencoder_features, labels)
                       ]

        results_fft.append(scores_lstm)

    results_fft = np.array(results_fft)

    mean_values = np.mean(results_fft, axis=0)

    print(f"FFT - final - {on_type} -> ARI - {mean_values[0]}")
    print(f"FFT - final - {on_type} -> AMI - {mean_values[1]}")
    print(f"FFT - final - {on_type} -> Hom - {mean_values[2]}")
    print(f"FFT - final - {on_type} -> Com - {mean_values[3]}")
    print(f"FFT - final - {on_type} -> VM - {mean_values[4]}")
    print(f"FFT - final - {on_type} -> CHS - {mean_values[5]}")
    print(f"FFT - final - {on_type} -> DBS - {mean_values[6]}")
    print(f"FFT - final - {on_type} -> SS - {mean_values[7]}")


def main_decision_tree(program):
    range_min = 1
    range_max = 96
    epochs = 100
    align = 2
    spike_verif_path = f'./figures/autoencoder_decisiontree/spike_verif'
    plot_path = f'./figures/autoencoder_decisiontree/'
    weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_decisiontree'

    if not os.path.exists(spike_verif_path):
        os.makedirs(spike_verif_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    spikes, labels = dsa.stack_simulations_range(range_min, range_max, True, True, alignment=align)

    autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                   encoder_layer_sizes=autoencoder_layer_sizes,
                                   decoder_layer_sizes=autoencoder_layer_sizes,
                                   code_size=autoencoder_code_size)

    encoder, autoenc = autoencoder.return_encoder()

    if program == "train":
        autoencoder.train(spikes, epochs=epochs)
        autoencoder.save_weights(weights_path)

        verify_random_outputs(spikes, encoder, autoenc, 10, path=spike_verif_path)

    if program == "test":
        autoencoder.load_weights(weights_path)

        codes_list = []
        labels_list = []

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                continue

            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=align)

            spikes = spikes[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes, encoder)

            codes_list.append(autoencoder_features)
            labels_list.append(labels)

        codes = np.vstack(codes_list)
        labels = np.hstack(labels_list)

        clf = DecisionTreeClassifier()
        clf = clf.fit(codes, labels)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                continue

            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=align)

            autoencoder_features = get_codes(spikes, encoder)
            y_pred = clf.predict(autoencoder_features)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

            try:
                scatter_plot.plot('DT' + str(len(autoencoder_features)), autoencoder_features, y_pred,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_dtree')
            except KeyError:
                pass


def main_show_sim4():
    spikes, labels = ds.get_dataset_simulation(simNr=4, align_to_peak=2)

    spikes = spikes[labels != 0]
    labels = labels[labels != 0]

    unique_labels = np.unique(labels)

    plot_path = f'./figures/show_sim4/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    for label in unique_labels:
        plt.figure()
        labeled_spikes = spikes[labels == label]
        for i in range(0, len(labeled_spikes)):
            plt.plot(np.arange(len(labeled_spikes[i])), labeled_spikes[i])
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.title(f"{LABEL_COLOR_MAP[label]}({label}) Cluster")
        plt.savefig(plot_path+f"spikes_of_cluster{label}")


def main_autoencoder_expansion(program):
    range_min = 1
    range_max = 96
    epochs = 100
    alignment = 2
    no_noise = False
    spike_verif_path = f'./figures/expanded/c{autoencoder_expanded_code_size}/spike_verif'
    plot_path = f'./figures/expanded/c{autoencoder_expanded_code_size}/'
    weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_c{autoencoder_expanded_layer_sizes}_expanded'

    if not os.path.exists(spike_verif_path):
        os.makedirs(spike_verif_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    spikes, labels = dsa.stack_simulations_range(range_min, range_max, True, no_noise, alignment=alignment)

    autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                   encoder_layer_sizes=autoencoder_expanded_layer_sizes,
                                   decoder_layer_sizes=autoencoder_expanded_layer_sizes,
                                   code_size=autoencoder_expanded_code_size)

    encoder, autoenc = autoencoder.return_encoder()

    if program == "train":
        autoencoder.train(spikes, epochs=epochs)
        autoencoder.save_weights(weights_path)

        verify_output(spikes, encoder, autoenc, path=spike_verif_path)
        verify_random_outputs(spikes, encoder, autoenc, 10, path=spike_verif_path)

    if program == "test":
        autoencoder.load_weights(weights_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                continue

            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=alignment)

            if no_noise == True:
                spikes = spikes[labels != 0]
                labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

            pn = 25
            labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
            # try:
            #     scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
            #                            marker='o')
            #     plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
            # except KeyError:
            #     pass
    if program == "validate":
        autoencoder.load_weights(weights_path)

        results_pca = []
        results_lstm = []
        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                continue

            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=alignment)

            if no_noise == True:
                spikes = spikes[labels != 0]
                labels = labels[labels != 0]


            pca_2d = PCA(n_components=2)
            pca_features = pca_2d.fit_transform(spikes)

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            pn = 25
            clustering_lstm_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
            clustering_pca_labels = SBM.parallel(pca_features, pn, ccThreshold=5, version=2)

            hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_lstm_labels)

            results_lstm.append([adjusted_rand_score(labels, clustering_lstm_labels),
                                 adjusted_mutual_info_score(labels, clustering_lstm_labels),
                                 hom,
                                 com,
                                 vm,
                                 calinski_harabasz_score(autoencoder_features, labels),
                                 davies_bouldin_score(autoencoder_features, labels),
                                 silhouette_score(autoencoder_features, labels)
                                 ])

            hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_pca_labels)
            results_pca.append([adjusted_rand_score(labels, clustering_pca_labels),
                                adjusted_mutual_info_score(labels, clustering_pca_labels),
                                hom,
                                com,
                                vm,
                                calinski_harabasz_score(pca_features, labels),
                                davies_bouldin_score(pca_features, labels),
                                silhouette_score(pca_features, labels)
                                ])

        mean_values = np.mean(results_lstm, axis=0)

        print(f"EXP -> ARI - {mean_values[0]}")
        print(f"EXP -> AMI - {mean_values[1]}")
        print(f"EXP -> Hom - {mean_values[2]}")
        print(f"EXP -> Com - {mean_values[3]}")
        print(f"EXP -> VM - {mean_values[4]}")
        print(f"EXP -> CHS - {mean_values[5]}")
        print(f"EXP -> DBS - {mean_values[6]}")
        print(f"EXP -> SS - {mean_values[7]}")

        mean_values = np.mean(results_pca, axis=0)

        print(f"PCA -> ARI - {mean_values[0]}")
        print(f"PCA -> AMI - {mean_values[1]}")
        print(f"PCA -> Hom - {mean_values[2]}")
        print(f"PCA -> Com - {mean_values[3]}")
        print(f"PCA -> VM - {mean_values[4]}")
        print(f"PCA -> CHS - {mean_values[5]}")
        print(f"PCA -> DBS - {mean_values[6]}")
        print(f"PCA -> SS - {mean_values[7]}")


def test_sum():
    simulation_number = 4
    spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

    unique_labels = np.unique(labels)

    for label in unique_labels:
        print(f"Cluster{label}: {np.mean(np.mean(spikes[labels==label], axis=1))}")
        print(f"Cluster{label}: {np.amin(np.mean(spikes[labels==label], axis=1))}")
        print(f"Cluster{label}: {np.amax(np.mean(spikes[labels==label], axis=1))}")
        print()


def alignment_show():
    simulation_number = 14

    for alignment in [False, True, 2, 3]:
        spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=alignment)

        pca_2d = PCA(n_components=2)
        pca_features = pca_2d.fit_transform(spikes)

        scatter_plot.plot('GT' + str(len(pca_features)), pca_features, labels, marker='o')
        plt.savefig(f'./figures/sim14_pca_align{alignment}')

        plt.figure()
        for i in range(0, len(spikes)):
            plt.plot(np.arange(len(spikes[i])), spikes[i])
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        if alignment == False:
            plt.title(f"Spikes unaligned")
        elif alignment == True:
            plt.title(f"Spikes aligned to average")
        elif alignment == 2:
            plt.title(f"Spikes aligned to global maximum")
        elif alignment == 3:
            plt.title(f"Spikes aligned to global minimum")
        plt.savefig(f'./figures/sim14_spikes_align{alignment}')


def apply_kmeans(simulation_number):
    spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=True)

    pca_2d = PCA(n_components=2)
    pca_features = pca_2d.fit_transform(spikes)

    scatter_plot.plot('PCA on sim4', pca_features, labels, marker='o')
    plt.savefig(f'./figures/sim{simulation_number}_pca')

    kmeans = KMeans(n_clusters=5, random_state=0).fit(pca_features)

    scatter_plot.plot('K-Means on sim 4', pca_features, kmeans.labels_, marker='o')
    plt.savefig(f'./figures/sim{simulation_number}_kmeans')

    pn = 25
    labels = SBM.parallel(pca_features, pn, ccThreshold=5, version=2)

    scatter_plot.plot('SBM on sim 4', pca_features, labels, marker='o')
    plt.savefig(f'./figures/sim{simulation_number}_sbm')


def print_half(simulation_number):
    no_noise = False
    align = True
    normalize = False
    scale = False

    train_spikes, train_labels, test_spikes, test_labels = dsa.stack_simulations_split_train_test(4, 5, no_noise=no_noise, alignment=align, normalize=normalize, scale=scale)

    pca_2d = PCA(n_components=2)
    pca_features = pca_2d.fit_transform(test_spikes[0])

    scatter_plot.plot('PCA on sim4', pca_features, test_labels[0], marker='o')
    plt.savefig(f'./figures/sim{simulation_number}_half_pca')

# alignment_show()
# apply_kmeans(4)
# print_half(4)
# main("sbm_graph", sub="")
main("autoencoder", sub="")
# main("autoencoder_single_sim", sub="")
# main("lstm_single_sim", sub="")
# main("lstm", sub="train")
# main("lstm", sub="validation")

# main("autoencoder_sim_array", sub="train")
# main("autoencoder_sim_array", sub="test")
# main("autoencoder_sim_range", sub="train")
# main("autoencoder_sim_range", sub="test")
# main("autoencoder_sim_range", sub="pre")
# main("benchmark_autoencoder", sub="")
# main("pipeline_test", sub="")


# case, alignment, on_type
# for alignment in [True, False]:
#     for case in ["original", "padded", "rolled", "reduced"]:
#         for on_type in ["real", "imag", "magnitude"]:
#             # main_fft("train", case, alignment, on_type)
#             # main_fft("test", case, alignment, on_type)
#             main_fft("validation", case, alignment, on_type)

# main_fft("validation", "padded", True, "magnitude")
# main_fft("validation", "reduced", True, "magnitude")
# main_fft("validation", "rolled", False, "imag")
# main_fft("validation", "padded", False, "real")

# case, alignment, on_type, window_type
# for window_type in ["blackman", "gaussian"]:
#     for alignment in [True, False, 2, 3]:
#         for on_type in ["real", "imag", "magnitude", "power", "phase", "concatenated"]:
#             # main_fft_windowed("train", alignment, on_type, window_type)
#             # main_fft_windowed("test", alignment, on_type, window_type)
#             main_fft_windowed("validation", alignment, on_type, window_type)

# main_dpss("train")
# main_dpss("test")
# main_iterative_dpss("test")
# main_dpss("validation")

# main_dpss_cascade()
# main_dpss_cascade_validation()

# main("lstm", sub="train")
# main("lstm", sub="test")

# main("split_lstm", sub="train")
# main("split_lstm", sub="test")
# main("lstm_pca_check", sub="")

# test_scale_normalize()

# test_fft_windowed(window_type="gaussian")
# test_fft_windowed(window_type="blackman")




# main_ensemble_algebraic("validate", "mean")
# main_ensemble_algebraic("validate", "sum")
# main_ensemble_algebraic("validate", "prod")
# main_ensemble_algebraic("validate", "min")
# main_ensemble_algebraic("validate", "max")
# main_ensemble_stacked("")

# main_dpss("train")
# main_dpss("test")

# main_dpss_test("sim4")
# main_dpss_test("sim13")
# main_dpss_test("sim26")
# main_dpss_test("sim40")
# main_decision_tree("train")
# main_decision_tree("test")
# main_show_sim4()
# main_autoencoder_expansion("train")
# main_autoencoder_expansion("test")
# main_autoencoder_expansion("validate")

# main("pca")
# main("pca_selected")


def main_test_spike_comparison():
    sim_nr = 4
    spikes, labels = ds.get_dataset_simulation(simNr=sim_nr, align_to_peak=2)

    diff_spikes = []

    spike_average = np.mean(spikes, axis=0)

    zeros = np.zeros((79,))
    list = []
    list.append(np.full((20, ), 0.1))
    list.append(np.full((20, ), 1))
    list.append(np.full((20, ), 1))
    list.append(np.full((19, ), 0.1))
    weights = np.hstack(list)
    print(weights)
    for label in np.unique(labels):
        cluster_spikes = spikes[labels == label]
        cluster_average = np.mean(cluster_spikes, axis=0)

        mse = (((cluster_spikes - cluster_average) ** 2) * weights).sum(axis=1)
        print(len(mse))
        mse_value = np.mean(mse)

        print(f"Cluster {label} error min    = {np.amin(mse):.2f}")
        print(f"Cluster {label} error max    = {np.amax(mse):.2f}")
        print(f"Cluster {label} error        = {mse_value:.2f}")

        mse = ((cluster_spikes - zeros) ** 2).sum(axis=1)
        mse_value = np.mean(mse)
        print(f"Cluster {label} error(0) min = {np.amin(mse):.2f}")
        print(f"Cluster {label} error(0) max = {np.amax(mse):.2f}")
        print(f"Cluster {label} error(0)     = {mse_value:.2f}")

        mse = ((cluster_spikes - spike_average) ** 2).sum(axis=1)
        mse_value = np.mean(mse)
        print(f"Cluster {label} error(A) min = {np.amin(mse):.2f}")
        print(f"Cluster {label} error(A) max = {np.amax(mse):.2f}")
        print(f"Cluster {label} error(A)     = {mse_value:.2f}")

        diff_spikes.append(cluster_average*weights)

    cluster_errors = []
    for i in range(len(diff_spikes)):
        matrix_row = []
        for j in range(len(diff_spikes)):
            mse = ((diff_spikes[i] - diff_spikes[j]) ** 2).sum()
            matrix_row.append(mse)
        cluster_errors.append(matrix_row)

    cluster_errors = np.array(cluster_errors)
    print(cluster_errors)


def main_spike_comparison():
    sim_nr = 4
    spikes, labels = ds.get_dataset_simulation(simNr=sim_nr, align_to_peak=2)

    spikes = spikes[labels != 0]
    labels = labels[labels != 0]

    cluster_averages = []
    cluster_averages.append(spikes[0])
    for i in range(1, len(spikes)):
        for j in range(len(cluster_averages)):
            if ((cluster_averages[j] - spikes[i]) ** 2).sum() > 2.5:
                cluster_averages.append(spikes[i])
            else:
                cluster_averages[j] = (cluster_averages[j] + spikes[i]) / 2

    print(len(cluster_averages))


def transform_spikes(spikes):
    spikes_shape = []
    for spike in spikes:
        spikes_shape.append((spike[0:len(spike)-5:5] - spike[5:len(spike):5] < 0) + 0)

    return np.array(spikes_shape)


def main_test_spike_comparison2():
    sim_nr = 4
    spikes, labels = ds.get_dataset_simulation(simNr=sim_nr, align_to_peak=2)

    spikes_shape = transform_spikes(spikes)

    for label in np.unique(labels):
        cluster_shapes = spikes_shape[labels == label]
        cluster_average = np.mean(cluster_shapes, axis=0)
        cluster_average = (cluster_average > 0.5) + 0

        mse = ((cluster_shapes - cluster_average) ** 2).sum(axis=1)

        print(len(mse))
        mse_value = np.mean(mse)
        print(cluster_shapes[0])
        print(cluster_average)
        print(f"Cluster {label} error min    = {np.amin(mse):.2f}")
        print(f"Cluster {label} error max    = {np.amax(mse):.2f}")
        print(f"Cluster {label} error        = {mse_value:.2f}")


# main_spike_comparison()
# main_test_spike_comparison()
# main_test_spike_comparison2()

# main("autoencoder_selected", sub="train")
# main("autoencoder_selected", sub="test")
# main("benchmark_lstm", sub="")
# main("benchmark_lstm_mul", sub="")
# main("benchmark_autoencoder", sub="")
# main("benchmark_pca", sub="")
# main("show_spike_shapes", sub="")

# main("autoencoder_sim_range", sub="test")
















