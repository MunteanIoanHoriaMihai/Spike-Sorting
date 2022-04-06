import csv

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from sklearn.decomposition import PCA

from utils.benchmark import benchmark_data as bd
from utils import constants as cs, scatter_plot as sp
from utils.dataset_parsing import simulations_dataset as ds
from feature_extraction import derivatives as deriv


def generate_dpss_windows(w_nr=4, plot=False, plot_s=False):
    M = 79  # 512
    NW = 2.5  # 2.5
    win, eigvals = signal.windows.dpss(M, NW, w_nr, return_ratios=True)
    if plot:
        fig, ax = plt.subplots(1)
        ax.plot(win.T, linewidth=1.)
        ax.set(xlim=[0, M - 1], ylim=[-0.21, 0.21], xlabel='Samples',
               title='DPSS, M=%d, NW=%0.1f' % (M, NW))
        ax.legend(['win[%d] (%0.4f)' % (ii, ratio)
                   for ii, ratio in enumerate(eigvals)])
        fig.tight_layout()
        plt.show()
    if plot_s:
        fig = plt.figure(figsize=(7, 10))
        # fig = plt.figure(figsize=(5, 7))
        axes = fig.subplots(w_nr)
        for i, ax in enumerate(axes):
            ax.set_title("DPSS window %d" % i)
            ax.plot(win[i].T, linewidth=2.)
            ax.set_xlabel("Time")
            ax.set_ylabel("Magnitude")
            plt.tight_layout()
        plt.show()
    return win


def stft_with_dpss_windows(sim_nr=1, w_nr=0):
    win = generate_dpss_windows()
    # w_nr = 0
    # sim_nr = 40
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window=win[w_nr], fs=1, nperseg=79)
    amplitude = np.abs(Zxx)
    amplitude_concat = amplitude.reshape(*amplitude.shape[:1], -1)
    pca_2d = PCA(n_components=2)
    amplitude_pca = pca_2d.fit_transform(amplitude_concat)
    sp.plot("STFT with dpss w%d" % w_nr, amplitude_pca, labels, marker='o')
    plt.savefig('figures/stft_plots/dpss_%d_sim_%d' % (w_nr, sim_nr))
    plt.show()


def save_all_dpss():
    pca_2d = PCA(n_components=2)
    win = generate_dpss_windows()
    w_nr = 3
    for alg_nr in range(2, 3):
        average = [0, 0, 0, 0, 0]
        # average = [0, 0, 0, 0, 0, 0]
        simulation_counter = 0
        for sim_nr in range(1, 96):
            if sim_nr != 25 and sim_nr != 27 and sim_nr != 44:
                spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2,
                                                           normalize_spike=False)
                sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window=win[w_nr], fs=1, nperseg=79)
                amplitude = np.abs(Zxx)
                amplitude = amplitude.reshape(*amplitude.shape[:1], -1)
                signal_pca = pca_2d.fit_transform(amplitude)
                sp.plot("Sim %d STFT with dpss w%d" % (sim_nr, w_nr), signal_pca, labels, marker='o')
                plt.savefig('figures/dpss/sim_%d_dpss_%d' % (sim_nr, w_nr))
                plt.show()
                alg_labels = bd.apply_algorithm(signal_pca, labels, alg_nr)
                results = bd.benchmark_algorithm_labeled_data(labels, alg_labels)
                # results = bd.benchmark_algorithm_extra(alg_labels, labels)
                simulation_counter += 1
                average += results
                formatted = ["%.3f" % number for number in results]
                row = [sim_nr, formatted[0], formatted[1], formatted[2], formatted[3], formatted[4]]
                # row = [sim_nr, formatted[0], formatted[1], formatted[2], formatted[3], formatted[4], formatted[5]]
                with open('./results/all_%s_dpss_%d.csv' % (cs.algorithms[alg_nr], w_nr), 'a+', newline='') as file:
                    writer = csv.writer(file, delimiter=',')
                    writer.writerow(row)
        average = average / simulation_counter
        with open('./results/all_%s_dpss_%d.csv' % (cs.algorithms[alg_nr], w_nr), 'a+', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(average)


def plot_spike_dpss():
    sim_nr = 15
    fs = 1
    w = generate_dpss_windows()
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    f0, t0, zxx0 = signal.stft(spikes, window=w[0], nperseg=79, fs=fs)
    f1, t1, zxx1 = signal.stft(spikes, window=w[1], nperseg=79, fs=fs)
    f2, t2, zxx2 = signal.stft(spikes, window=w[2], nperseg=79, fs=fs)
    f3, t3, zxx3 = signal.stft(spikes, window=w[3], nperseg=79, fs=fs)
    Zxx = np.concatenate([zxx0, zxx1, zxx2, zxx3], axis=2)
    t = np.concatenate([t0, t1, t2, t3])
    f = np.concatenate([f0, f1, f2, f3])
    stft_signal = Zxx.reshape(*Zxx.shape[:1], -1)
    print(Zxx.shape)
    print(stft_signal.shape)

    spike_nr = 100
    # plt.plot(stft_signal)  # this is interesting
    # plt.plot(stft_signal[spike_nr])
    # plt.show()
    # plt.plot(np.arange(79), spikes[spike_nr])
    # plt.show()
    # plt.plot(np.abs(Zxx)[spike_nr])
    # plt.show()
    s = plt.imshow(np.abs(Zxx)[spike_nr], cmap='jet', aspect='auto', origin='lower')
    plt.colorbar(s)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title("Amplitude dpss fs_%.2f nperseg_%d" % (fs, 79))
    # plt.savefig('./figures/stft_plots/amplitude_fs_%d_nperseg_%d' % (fs, nperseg))
    plt.show()


def stft_dpss_concat(spikes, derivatives=False):
    fs = 1
    w = generate_dpss_windows()
    f0, t0, zxx0 = signal.stft(spikes, window=w[0], nperseg=79, fs=fs)
    f1, t1, zxx1 = signal.stft(spikes, window=w[1], nperseg=79, fs=fs)
    f2, t2, zxx2 = signal.stft(spikes, window=w[2], nperseg=79, fs=fs)
    f3, t3, zxx3 = signal.stft(spikes, window=w[3], nperseg=79, fs=fs)
    if derivatives:
        zxx0 = np.apply_along_axis(deriv.compute_fdmethod_1spike, 1, np.abs(zxx0))
        zxx1 = np.apply_along_axis(deriv.compute_fdmethod_1spike, 1, np.abs(zxx1))
        zxx2 = np.apply_along_axis(deriv.compute_fdmethod_1spike, 1, np.abs(zxx2))
        zxx3 = np.apply_along_axis(deriv.compute_fdmethod_1spike, 1, np.abs(zxx3))
    Zxx = np.concatenate([zxx0, zxx2], axis=2)
    t = np.concatenate([t0, t1, t2, t3])
    f = np.concatenate([f0, f1, f2, f3])
    stft_signal = Zxx.reshape(*Zxx.shape[:1], -1)
    print(Zxx.shape)
    print(stft_signal.shape)
    return np.array(stft_signal)


def feature_extraction_stft_dpss_concat(sim_nr=79, plot_gt=False, plot_sbm=False):
    pca2d = PCA(n_components=2)
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    signal_dpss = stft_dpss_concat(spikes)
    amplitude = np.abs(signal_dpss)
    signal_pca = pca2d.fit_transform(amplitude)
    if plot_gt:
        sp.plot("Sim %d STFT dpss concat w02" % sim_nr, signal_pca, labels, marker='o')
        plt.savefig('figures/dpss/sim_%d_dpss_c_02' % sim_nr)
        plt.show()

    sbm_labels = bd.apply_algorithm(signal_pca, labels, 2)
    if plot_sbm:
        sp.plot("SBM Sim %d STFT dpss concat w02" % sim_nr, signal_pca, sbm_labels, marker='o')
        plt.savefig('figures/dpss/sbm_sim_%d_dpss_c_02' % sim_nr)
        plt.show()

    sbm_results = bd.benchmark_algorithm_labeled_data(labels, sbm_labels)
    return sbm_results


def loop_stft_dpss_concat():
    """
        Writes the result of the stft with dpss into csv files.
        Computes the average.
    :return:
    """
    average = [0, 0, 0, 0, 0]
    simulation_counter = 0

    for i in range(1, 96):
        if i != 25 and i != 27 and i != 44:
            simulation_counter += 1
            results = feature_extraction_stft_dpss_concat(sim_nr=i, plot_gt=True, plot_sbm=True)
            average += results
            formatted = ["%.3f" % number for number in results]
            row = [i, formatted[0], formatted[1], formatted[2], formatted[3], formatted[4]]

            with open('./results/dpss_stft_concat_2_02.csv', 'a+', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(row)
    average = average / simulation_counter
    with open('./results/dpss_stft_concat_2_02.csv', 'a+', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(average)


# generate_stft_windows(w_nr=4, plot=True)
# stft_with_dpss_windows(sim_nr=37, w_nr=0)
# save_all_dpss()
# plot_spike_dpss()
# feature_extraction_stft_dpss_concat(sim_nr=1)
# loop_stft_dpss_concat()
