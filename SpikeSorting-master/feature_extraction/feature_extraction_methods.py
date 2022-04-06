import pickle

import numpy as np
import scipy.signal as signal
from PyEMD import EMD
from scipy.fftpack import fft
from scipy.signal import hilbert
from scipy.stats import kstest, norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from feature_extraction.wlt import discretewlt as dwt, wavelets as wlt
import libraries.SimpSOM as sps
from feature_extraction import shape_features
import feature_extraction.derivatives as deriv
from feature_extraction.fourier import stft_dpss
from feature_extraction.slt import superlets as slt


def continuous_wavelet_transform(spikes):
    """
    Load the dataset after applying continuous wavelets on 2 dimensions
    :returns result_spikes: matrix - the 2-dimensional points resulted
    """
    cwt_features = wlt.fd_wavelets(spikes, derivatives=True)

    # scaler = StandardScaler()
    # features = scaler.fit_transform(cwt_features)

    return cwt_features


def discrete_wavelet_transform(spikes):
    """
    Load the dataset after dwt on 2 dimensions

    :returns result_spikes: matrix - the 2-dimensional points resulted
    """
    dwt_features = dwt.dwt_fd_method(spikes)
    # scaler = StandardScaler()
    # features = scaler.fit_transform(dwt_features)

    return dwt_features


def discrete_wavelet_transform_ks(spikes):
    """
    Load the dataset after dwt on 2 dimensions

    :returns result_spikes: matrix - the 2-dimensional points resulted
    """
    dwt_features = dwt.compute_haar_ks(spikes)

    return dwt_features


def superlets(spikes):
    slt_features = slt.slt(spikes, 2, 1.1)
    # slt_features = slt.slt(spikes, 5, 1.8)

    scaler = StandardScaler()
    features = scaler.fit_transform(slt_features)

    return features


def derivatives2d(spikes):
    """
    Extract derivatives on 2 dimensions
    :returns result_spikes: matrix - the 2-dimensional points resulted
    """
    derivative_features = deriv.compute_fdmethod(spikes)

    scaler = StandardScaler()
    features = scaler.fit_transform(derivative_features)

    return features


def derivatives3d(spikes):
    result_spikes = deriv.compute_fdmethod3d(spikes)

    return result_spikes


def fsde(spikes):
    result_spikes = deriv.fsde_method(spikes)

    return result_spikes


def shape_phase_distribution(spikes):
    features = shape_features.get_shape_phase_distribution_features(spikes)

    return features


def hilbert_envelope(spikes):
    spikes_hilbert = hilbert(spikes)
    envelope = np.abs(spikes_hilbert)

    return envelope


def hht(spikes):
    emd = EMD()
    spikes = np.array(spikes)
    features = np.zeros((spikes.shape[0], 3))
    for i, spike in enumerate(spikes):
        emd(spike)
        IMFs, res = emd.get_imfs_and_residue()

        hb = hilbert(IMFs)
        phase = np.unwrap(np.angle(hb))
        inst_f = (np.diff(phase) / (2.0 * np.pi))

        # time = np.arange(78)
        # fig = plt.figure(figsize=(5, 7))
        # axes = fig.subplots(IMFs.shape[0])
        # for imf, ax in enumerate(axes):
        #     ax.set_title("Instantaneous frequency of IMF%s" % imf)
        #     ax.plot(time, inst_f[imf])
        #     ax.set_xlabel("Time")
        #     ax.set_ylabel("Magnitude")
        #     plt.tight_layout()
        #     plt.savefig('figures/EMD/sim' + str() + '_spike' + str(i) + '_inst_freq_on_IMFs' + '.png')
        # plt.show()

        features[i] = np.array([np.max(spike), np.max(inst_f), np.min(inst_f)])
        # f = np.ndarray.flatten(reduce_dimensionality(inst_f, method='derivatives2d'))
        # features[i][:f.shape[0]] = f

    return features


def hht_ks(spikes):
    emd = EMD()
    spikes = np.array(spikes)
    features = np.zeros((spikes.shape[0], 468))
    for i, spike in enumerate(spikes):
        emd(spike)
        IMFs, res = emd.get_imfs_and_residue()

        hb = hilbert(IMFs)
        phase = np.abs(hb)
        inst_f = phase
        # phase = np.unwrap(np.angle(hb))
        # inst_f = (np.diff(phase) / (2.0 * np.pi))

        concat_imfs = np.concatenate((inst_f[:]))

        features[i, 0:concat_imfs.shape[0]] = concat_imfs
        # f = np.ndarray.flatten(reduce_dimensionality(inst_f, method='derivatives2d'))
        # features[i][:f.shape[0]] = f

    # pca = PCA(n_components=78)
    # features = pca.fit_transform(features)

    # print(features.shape)

    k_s = []
    for i in range(468):
        coeffs = []
        for spike_coeff_pos in range(0, len(features)):
            coeffs.append(features[spike_coeff_pos][i])
        loc, scale = norm.fit(coeffs)
        # create a normal distribution with loc and scale
        n = norm(loc=loc, scale=scale)
        s, p = kstest(coeffs, n.cdf)
        k_s.append(p)
    min_10_pos = take_min_10positions(k_s)
    res = []
    for res_index in range(len(features)):
        lst = []
        for i in range(len(min_10_pos)):
            lst.append(features[res_index][i])
        res.append(lst)

    return np.array(res)


def take_min_10positions(coeff):
    ordered = coeff.copy()
    ordered.sort()
    result_pos = []
    for i in range(0, 10):
        result_pos.append(coeff.index(ordered[i]))
    return result_pos


def emd_signal_no_residuum(spikes):
    emd = EMD()

    spikes = np.array(spikes)
    features = np.zeros((spikes.shape[0], spikes.shape[1]))
    for i, spike in enumerate(spikes):
        emd(spike)
        IMFs, res = emd.get_imfs_and_residue()
        features[i] = np.sum(IMFs, axis=0)

    return features


def emd_imf_derivatives(spikes):
    emd = EMD()

    features = np.zeros((spikes.shape[0], 12))
    for i, spike in enumerate(spikes):
        emd(spike)
        IMFs, res = emd.get_imfs_and_residue()

        f = np.array(deriv.compute_fdmethod(IMFs))

        flattened_f = np.ndarray.flatten(f)
        features[i][0:flattened_f.shape[0]] = flattened_f

    return features


def kohonen_som(spikes, dim, epochs, learn_rate, title="", plot=True):
    filename = 'kohonen_' + str(dim[0]) + 'x' + str(dim[1]) + '_ep' + str(
        epochs) + '_' + title

    try:
        k_map = pickle.load(open('models/' + filename + '.sav', 'rb'))
    except FileNotFoundError:
        k_map = sps.somNet(dim[0], dim[1], spikes, PBC=True, PCI=True)
        k_map.train(startLearnRate=learn_rate, epochs=epochs)
        pickle.dump(k_map, open('models/' + filename + '.sav', 'wb'))

        if plot:
            k_map.diff_graph(show=True, printout=True, filename='figures/kohonen/' + filename)
    features = np.array(k_map.project(spikes, show=True, printout=True))

    return features


def stft(spikes):
    sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window='blackman', fs=1, nperseg=52)
    stft_signal = Zxx.reshape(*Zxx.shape[:1], -1)
    amplitude = np.abs(stft_signal)
    return amplitude


def stft_d(spikes):
    sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window='blackman', fs=1, nperseg=45)
    amplitude = np.abs(Zxx)
    amplitude = np.apply_along_axis(deriv.compute_fdmethod_1spike, 2, amplitude)
    amplitude = amplitude.reshape(*amplitude.shape[:1], -1)
    return amplitude


def stft_multitaper(spikes):
    win = stft_dpss.generate_dpss_windows()
    f0, t0, zxx0 = signal.stft(spikes, window=win[0], nperseg=79, fs=1)
    f2, t2, zxx2 = signal.stft(spikes, window=win[2], nperseg=79, fs=1)
    Zxx = np.concatenate([zxx0, zxx2], axis=2)
    stft_signal = Zxx.reshape(*Zxx.shape[:1], -1)
    amplitude = np.abs(stft_signal)
    return amplitude


def stft_multitaper_w(spikes):
    win = stft_dpss.generate_dpss_windows()
    f, t, Zxx = signal.stft(spikes, window=win[1], nperseg=58, fs=1)
    stft_signal = Zxx.reshape(*Zxx.shape[:1], -1)
    amplitude = np.abs(stft_signal)
    return amplitude


def fourier_real(spikes):
    fft_signal = fft(spikes)
    X = [x.real for x in fft_signal[:, 0:40]]
    return X


def fourier_imaginary(spikes):
    fft_signal = fft(spikes)
    Y = [x.imag for x in fft_signal[:, 0:40]]
    return Y


def fourier_amplitude(spikes):
    fft_signal = fft(spikes)
    X = [x.real for x in fft_signal[:, 0:40]]
    Y = [x.imag for x in fft_signal[:, 0:40]]
    amplitude = np.sqrt(np.add(np.multiply(X, X), np.multiply(Y, Y)))
    return amplitude


def fourier_phase(spikes):
    fft_signal = fft(spikes)
    X = [x.real for x in fft_signal[:, 0:40]]
    Y = [x.imag for x in fft_signal[:, 0:40]]
    phase = np.arctan2(Y, X)
    return phase


def fourier_power(spikes):
    fft_signal = fft(spikes)
    X = [x.real for x in fft_signal[:, 0:40]]
    Y = [x.imag for x in fft_signal[:, 0:40]]
    amplitude = np.sqrt(np.add(np.multiply(X, X), np.multiply(Y, Y)))
    power = amplitude * amplitude
    return power


def reduce_dimensionality(n_features, method='PCA2D'):
    if method.lower() == 'pca2d':
        pca_2d = PCA(n_components=2)
        features = pca_2d.fit_transform(n_features)
    elif method.lower() == 'pca3d':
        pca_3d = PCA(n_components=3)
        features = pca_3d.fit_transform(n_features)
    elif method.lower() == 'pca10d':
        pca_5d = PCA(n_components=10)
        features = pca_5d.fit_transform(n_features)
    elif method.lower() == 'derivatives2d':
        features = deriv.compute_fdmethod(n_features)
    elif method.lower() == 'derivatives3d':
        features = deriv.compute_fdmethod3d(n_features)
    elif method.lower() == 'derivatives_pca2d':
        features = deriv.compute_fdmethod(n_features)
        pca_2d = PCA(n_components=2)
        features = pca_2d.fit_transform(features)
    else:
        raise Exception('Dimensionality Reduction method unknown.')
    return features


def apply_feature_extraction_method(spikes, feature_extraction_method=None, dim_reduction_method=None, **kwargs):
    spikes = np.array(spikes)

    options = {
        'som_dim': [40, 40],
        'som_epochs': 6000,
        'som_learn_rate': 0.1,
        'title': "",
        'extra_plot': False,
    }
    options.update(kwargs)
    if feature_extraction_method.lower() == 'pca2d':
        features = reduce_dimensionality(spikes, feature_extraction_method)
    elif feature_extraction_method.lower() == 'pca3d':
        features = reduce_dimensionality(spikes, feature_extraction_method)
    elif feature_extraction_method.lower() == 'pca10d':
        features = reduce_dimensionality(spikes, feature_extraction_method)
    elif feature_extraction_method.lower() == 'derivatives2d':
        features = derivatives2d(spikes)
    elif feature_extraction_method.lower() == 'derivatives3d':
        features = derivatives3d(spikes)
    elif feature_extraction_method.lower() == 'fsde':
        features = fsde(spikes)
    elif feature_extraction_method.lower() == 'slt':
        features = superlets(spikes)
    elif feature_extraction_method.lower() == 'cwt':
        features = continuous_wavelet_transform(spikes)
    elif feature_extraction_method.lower() == 'dwt':
        features = discrete_wavelet_transform(spikes)
    elif feature_extraction_method.lower() == 'dwt_ks':
        features = discrete_wavelet_transform_ks(spikes)
    elif feature_extraction_method.lower() == 'hilbert':
        features = hilbert_envelope(spikes)
    elif feature_extraction_method.lower() == 'hht':
        features = hht(spikes)
    elif feature_extraction_method.lower() == 'hht_ks':
        features = hht_ks(spikes)
    elif feature_extraction_method.lower() == 'emd':
        features = emd_imf_derivatives(spikes)
    elif feature_extraction_method.lower() == 'som':
        features = kohonen_som(spikes, options['som_dim'], options['som_epochs'], options['som_learn_rate'],
                               options['title'], plot=options['extra_plot'])
    elif feature_extraction_method.lower() == 'stft':
        features = stft(spikes)
    elif feature_extraction_method.lower() == 'stft_d':
        features = stft_d(spikes)
    elif feature_extraction_method.lower() == 'stft_dpss':
        features = stft_multitaper(spikes)
    elif feature_extraction_method.lower() == 'fourier_real':
        features = fourier_real(spikes)
    elif feature_extraction_method.lower() == 'fourier_imaginary':
        features = fourier_imaginary(spikes)
    elif feature_extraction_method.lower() == 'fourier_amplitude':
        features = fourier_amplitude(spikes)
    elif feature_extraction_method.lower() == 'fourier_phase':
        features = fourier_phase(spikes)
    elif feature_extraction_method.lower() == 'fourier_power':
        features = fourier_power(spikes)
    elif feature_extraction_method.lower() == 'shape':
        features = shape_features.get_shape_phase_distribution_features(spikes)
    else:
        raise Exception('Feature extraction method unknown.')

    if dim_reduction_method is not None:
        features = reduce_dimensionality(features, dim_reduction_method)

    return features
