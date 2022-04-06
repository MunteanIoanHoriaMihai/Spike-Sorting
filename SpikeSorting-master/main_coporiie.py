import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.fftpack import fft

from feature_extraction.autoencoder import fft_input
from utils.sbm import SBM
from utils.dataset_parsing import simulations_dataset as ds
from utils import scatter_plot
from utils.constants import autoencoder_layer_sizes, autoencoder_code_size
from feature_extraction.autoencoder.model_auxiliaries import verify_output, get_codes, verify_random_outputs
from feature_extraction.autoencoder.autoencoder import AutoencoderModel


def main(program, sub):
    if program == "simple_pad":
        spikes, labels = ds.get_dataset_simulation(1, align_to_peak=True)




        """
        PADDING AT END WITH 0 and moving before max-amplitude to end (WITH-OUT-ALIGNMENT)
        """
        # spikes = np.pad(spikes, ((0, 0), (0, 128 - len(spikes[0]))), 'constant')
        # peak_ind = np.argmax(spikes, axis=1)
        #
        # spikes = [np.roll(spikes[i], -peak_ind[i]) for i in range(len(spikes))]
        # spikes = np.array(spikes)
        #
        # fft_signal = fft_test(spikes)
        #
        # fft_real = [x.real for x in fft_signal[0]]
        # fft_imag = [x.imag for x in fft_signal[0]]
        #
        # plt.plot(np.arange(len(spikes[0])), spikes[0])
        # plt.title(f"padded spike")
        # plt.savefig(f'figures/autoencoder/fft_test/rolled_woA_spike')
        # plt.cla()
        #
        # plt.plot(np.arange(len(fft_real)), fft_real)
        # plt.title(f"FFT real part")
        # plt.savefig(f'figures/autoencoder/fft_test/rolled_woA_fft_real')
        # plt.cla()
        #
        # plt.plot(np.arange(len(fft_imag)), fft_imag)
        # plt.title(f"FFT imag part")
        # plt.savefig(f'figures/autoencoder/fft_test/rolled_woA_fft_imag')
        # plt.cla()

        """
        PADDING AT END WITH 0 and moving before max-amplitude to end (WITH-ALIGNMENT)
        """
        # spikes = np.pad(spikes, ((0, 0), (0, 128 - len(spikes[0]))), 'constant')
        # peak_ind = np.argmax(spikes, axis=1)
        #
        # spikes = [np.roll(spikes[i], -peak_ind[i]) for i in range(len(spikes))]
        # spikes = np.array(spikes)
        #
        # fft_signal = fft_test(spikes)
        #
        # fft_real = [x.real for x in fft_signal[0]]
        # fft_imag = [x.imag for x in fft_signal[0]]
        #
        # plt.plot(np.arange(len(spikes[0])), spikes[0])
        # plt.title(f"padded spike")
        # plt.savefig(f'figures/autoencoder/fft_test/rolled_wA_spike')
        # plt.cla()
        #
        # plt.plot(np.arange(len(fft_real)), fft_real)
        # plt.title(f"FFT real part")
        # plt.savefig(f'figures/autoencoder/fft_test/rolled_wA_fft_real')
        # plt.cla()

        """
        REDUCING SPIKE TO 64 (WITH-OUT-ALIGNMENT)
        """
        spikes = [spike[0:64] for spike in spikes]
        spikes = np.array(spikes)

        fft_signal = fft(spikes)

        fft_real = [x.real for x in fft_signal[0]]
        fft_imag = [x.imag for x in fft_signal[0]]

        plt.plot(np.arange(len(spikes[0])), spikes[0])
        plt.title(f"padded spike")
        plt.savefig(f'figures/autoencoder/fft/reduced_woA_spike')
        plt.cla()

        plt.plot(np.arange(len(fft_real)), fft_real)
        plt.title(f"FFT real part")
        plt.savefig(f'figures/autoencoder/fft/reduced_woA_fft_real')
        plt.cla()

        plt.plot(np.arange(len(fft_imag)), fft_imag)
        plt.title(f"FFT imag part")
        plt.savefig(f'figures/autoencoder/fft/reduced_woA_fft_imag')
        plt.cla()

        """
        REDUCING SPIKE TO 64 (WITH-ALIGNMENT)
        """
        spikes = [spike[0:64] for spike in spikes]
        spikes = np.array(spikes)

        fft_signal = fft(spikes)

        fft_real = [x.real for x in fft_signal[0]]
        fft_imag = [x.imag for x in fft_signal[0]]

        plt.plot(np.arange(len(spikes[0])), spikes[0])
        plt.title(f"padded spike")
        plt.savefig(f'figures/autoencoder/fft/reduced_wA_spike')
        plt.cla()

        plt.plot(np.arange(len(fft_real)), fft_real)
        plt.title(f"FFT real part")
        plt.savefig(f'figures/autoencoder/fft/reduced_wA_fft_real')
        plt.cla()

        plt.plot(np.arange(len(fft_imag)), fft_imag)
        plt.title(f"FFT imag part")
        plt.savefig(f'figures/autoencoder/fft/reduced_wA_fft_imag')
        plt.cla()


    elif program == "run":
        range_min = 1
        range_max = 96
        epochs = 100
        case = "reduced"
        alignment = False
        on_type = "real"
        spike_verif_path = f'./figures/fft/c{autoencoder_code_size}/{on_type}/{"wA" if alignment else "woA"}/{case}/spike_verif'
        plot_path = f'./figures/fft/c{autoencoder_code_size}/{on_type}/{"wA" if alignment else "woA"}/{case}/'
        weights_path = f'feature_extraction/autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft-{on_type}-{case}_{"wA" if alignment else "woA"}'

        if not os.path.exists(spike_verif_path):
            os.makedirs(spike_verif_path)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        fft_real, fft_imag = fft_input.apply_fft_on_range(case, alignment, range_min, range_max)

        if on_type == "real":
            spikes = fft_real
        elif on_type == "imag":
            spikes = fft_imag
        elif on_type == "magnitude":
            spikes = np.sqrt(fft_real*fft_real + fft_imag*fft_imag)

        spikes = np.array(spikes)

        autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                       encoder_layer_sizes=autoencoder_layer_sizes,
                                       decoder_layer_sizes=autoencoder_layer_sizes,
                                       code_size=autoencoder_code_size)

        encoder, autoenc = autoencoder.return_encoder()

        if sub == "train":
            autoencoder.train(spikes, epochs=epochs)
            autoencoder.save_weights(weights_path)

            verify_output(spikes, encoder, autoenc, path=spike_verif_path)
            verify_random_outputs(spikes, encoder, autoenc, 10, path=spike_verif_path)

        if sub == "test":
            autoencoder.save_weights(weights_path)

            for simulation_number in range(range_min, range_max):
                if simulation_number == 25 or simulation_number == 44:
                    continue

                fft_real, fft_imag, labels = fft_input.apply_fft_on_sim(sim_nr=simulation_number, case=case, alignment=alignment)
                if on_type == "real":
                    spikes = fft_real
                elif on_type == "imag":
                    spikes = fft_imag
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

                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')



main(program="run", sub="train")
main(program="run", sub="test")
