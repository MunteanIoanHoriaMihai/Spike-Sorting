import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import iqr, skew

from feature_extraction import derivatives


def get_closest_index(signal, index, target):
    """Returns either index or index + 1, the one that is closest to target"""
    return index if abs(signal[index] - target) < abs(signal[index + 1] - target) else index + 1


def get_max(signal):
    max_index = np.argmax(signal)
    max_value = signal[max_index]
    return max_value, max_index


def get_min(signal):
    min_index = np.argmin(signal)
    min_value = signal[min_index]
    return min_value, min_index


def get_half_width(spike):
    left_width_index = 0
    right_width_index = 0

    spike_max, spike_max_index = get_max(spike)

    for index in range(spike_max_index, len(spike) - 1):
        if spike[index] > spike_max / 2 > spike[index + 1]:
            right_width_index = get_closest_index(spike, index, spike_max)
            break
    for index in range(spike_max_index, 0, -1):
        if spike[index] < spike_max / 2 < spike[index + 1]:
            left_width_index = get_closest_index(spike, index, spike_max)
            break

    spike_half_width = right_width_index - left_width_index
    return spike_half_width, right_width_index, left_width_index


def get_derivatives(signal):
    fd = np.array(derivatives.compute_derivative5stencil(signal))
    sd = np.array(derivatives.compute_derivative5stencil(fd))
    return fd, sd


def get_valleys_near_peak(spike):
    spike_max, spike_max_index = get_max(spike)
    try:
        left_min_index = 1 + argrelextrema(spike[1:spike_max_index], np.less)[0][0]
    except IndexError:
        left_min_index = 0
    try:
        right_min_index = spike_max_index + argrelextrema(spike[spike_max_index:], np.less)[0][0]
    except IndexError:
        right_min_index = len(spike) - 1
    return left_min_index, spike[left_min_index], right_min_index, spike[right_min_index]


def get_shape_phase_distribution_features(spikes, plot=False):
    """
    :returns derivative based features
    P1	First zero-crossing of the FD before the action potential has been detected
    P2	Valley of the FD of the action potential
    P3	Second zero-crossing of the FD of the action potential that has been detected
    P4	Peak of the FD of the action potential
    P5	Third zero-crossing of the FD after the action potential has been detected
    P6	Valley of the FD after the action potential
    """
    features = []
    p1_fd_min_before_peak_index = 0

    for spike in spikes:
        fd, sd = get_derivatives(spike)

        spike_max, spike_max_index = get_max(spike)
        # spike_min, spike_min_index = get_min(spike)

        fd_max, fd_max_index = get_max(fd)
        fd_min, fd_min_index = get_min(fd)

        sd_max, sd_max_index = get_max(sd)
        sd_min, sd_min_index = get_min(sd)

        # p2_argmin_fd = np.argmin(fd)

        # for index in range(p2_argmin_fd - 1, 0, -1):
        #     if fd[index] > 0 > fd[index + 1]:
        #         p1_fd_min_before_peak_index = get_closest_index(fd, index, 0)
        #         break
        # if p1_fd_min_before_peak_index == 0:
        #     p1_fd_min_before_peak_index = 2

        p3_sd_max_index = sd_max_index
        p5_sd_min_index = sd_min_index
        p4_fd_max_index = fd_max_index
        # p6_fd_min_after_p5_index = np.argmin(fd[p5_sd_min_index:p5_sd_min_index + 20]) + p5_sd_min_index

        # f1 = p5_sd_min_index - p1_fd_min_before_peak_index
        # f2 = fd[p4_fd_max_index] - fd[p2_argmin_fd]
        # f3 = fd[p6_fd_min_after_p5_index] - fd[p2_argmin_fd]
        #
        # f5 = math.log2(abs((fd[p4_fd_max_index] - fd[p2_argmin_fd]) / (p4_fd_max_index - p2_argmin_fd)))
        # f6 = (fd[p6_fd_min_after_p5_index] - fd[p4_fd_max_index]) / (p6_fd_min_after_p5_index - p4_fd_max_index)
        # f7 = math.log2(
        #     abs((fd[p6_fd_min_after_p5_index] - fd[p2_argmin_fd]) / (p6_fd_min_after_p5_index - p2_argmin_fd)))
        # f8_rms_pre_peak = np.sqrt(np.mean(fd[:spike_max_index] * fd[:spike_max_index]))
        # if math.isnan(f8_rms_pre_peak):
        #     f8_rms_pre_peak = 0
        # f9 = ((fd[p2_argmin_fd] - fd[p1_fd_min_before_peak_index]) / (p2_argmin_fd - p1_fd_min_before_peak_index)) / (
        #         (fd[p3_sd_max_index] - fd[p2_argmin_fd]) / (p3_sd_max_index - p2_argmin_fd))
        # f10 = ((fd[p4_fd_max_index] - fd[p3_sd_max_index]) / (p4_fd_max_index - p3_sd_max_index)) / (
        #         (fd[p5_sd_min_index] - fd[p4_fd_max_index]) / (p5_sd_min_index - p4_fd_max_index))
        # f11 = fd[p2_argmin_fd] / fd[p4_fd_max_index]
        # f12 = fd[p1_fd_min_before_peak_index]
        # f13 = fd[p3_sd_max_index]
        f14_fd_max = fd[p4_fd_max_index]
        # f15 = fd[p5_sd_min_index]
        # f16 = fd[p6_fd_min_after_p5_index]
        # f17 = sd[p1_fd_min_before_peak_index]
        f18_sd_max = sd[p3_sd_max_index]
        f19_sd_min = sd[p5_sd_min_index]
        f20 = iqr(fd)
        f21 = iqr(sd)
        # f22 = kurtosis(fd)
        # f23 = skew(fd)
        # f24 = skew(sd)

        # spike_half_width, right_width_index, left_width_index = get_half_width(spike)

        if plot:
            plt.plot(np.arange(79), spike, label='spike')
            # plt.plot(np.arange(79), fd, '--', label='first derivative')
            # plt.plot(np.arange(79), sd, ':', label='second derivative')
            # plt.plot(spike_max_index, spike_max, marker='o', label='spike peak')
            # plt.plot(p1_fd_min_before_peak_index, fd[p1_fd_min_before_peak_index], marker='o',
            #          label='P1: FD valley before peak')
            # plt.plot(p2_argmin_fd, fd[p2_argmin_fd], marker='o', label='P2: FD valley')
            # plt.plot(p3_sd_max_index, f18_sd_max, marker='o', label='P3: SD peak')
            # plt.plot(p4_fd_max_index, fd[p4_fd_max_index], marker='o', label='P4: FD peak')
            # plt.plot(p5_sd_min_index, f19_sd_min, marker='o', label='SD valley')
            # plt.plot(p6_fd_min_after_p5_index, fd[p6_fd_min_after_p5_index], marker='o', label='P6: FD valley after P5')
            # plt.axvline(x=spike_max_index)
            # plt.plot([0, 80], [0, 0])
            # plt.title("Shape and Phase Features")
            # plt.plot(left_width_index + np.arange(spike_half_width),
            #          np.repeat((spike[left_width_index] + spike[right_width_index]) / 2, spike_half_width),
            #          label='Half-Width')
            # plt.legend()
            # plt.title("Spike Half-Width")
            # plt.xlabel("Time")
            # plt.ylabel("Amplitude")
            # plt.show()

        features.append(
            # [f14_fd_max - fd_min, f18_sd_max, f18_sd_max, spike_max,]
            [f14_fd_max, fd_min, f18_sd_max, f19_sd_min, spike_max, f20, f21]
            # [spike_max, f14_fd_max, fd_min, spike_half_width]
            # [spike_max, f19_sd_min, rms, ]
        )

    return np.array(features)


def describe_spike(spike):
    spike_max, spike_max_index = get_max(spike)
    spike_min, spike_min_index = get_min(spike)
    spike_half_width, right_width_index, left_width_index = get_half_width(spike)
    left_min_index, left_min, right_min_index, right_min = get_valleys_near_peak(spike)
    valley_time_diff = abs(right_min_index - left_min_index)
    spike_mean = np.mean(spike)
    spike_iqr = iqr(spike)
    spike_skew = skew(spike)
    spike_rms = np.sqrt(np.mean(spike * spike))
    # sum_before_spike = np.sum(spike[:spike_max])
    # sum_after_spike = np.sum(spike[spike_max:])
    sum_before_first_valley = np.sum(spike[:left_min_index])
    sum_after_second_valley = np.sum(spike[right_min_index:])

    fd, sd = get_derivatives(spike)
    fd_max, fd_max_index = get_max(fd)
    fd_min, fd_min_index = get_min(fd)
    sd_max, sd_max_index = get_max(sd)
    sd_min, sd_min_index = get_min(sd)

    features = {
        "spike_max_index": spike_max_index,
        "spike_max": spike_max,
        "spike_min_index": spike_min_index,
        "spike_min": spike_min,
        "spike_half_width": spike_half_width,
        "valley_time_diff": valley_time_diff,
        # "sum_before_spike": sum_before_spike,
        # "sum_after_spike": sum_after_spike,
        "sum_before_first_valley": sum_before_first_valley,
        "sum_after_second_valley": sum_after_second_valley,
        "spike_mean": spike_mean,
        "spike_iqr": spike_iqr,
        "spike_skew": spike_skew,
        "spike_rms": spike_rms,
        "fd_max_index": fd_max_index,
        "fd_max": fd_max,
        "fd_min_index": fd_min_index,
        "fd_min": fd_min,
    }
    return features


def print_features(features):
    for key, value in features.items():
        print("{:<25} {:.3f}".format(key, value))


def describe_cluster(cluster):
    sum_features = {}
    for spike in cluster:
        spike_features = describe_spike(spike)
        for key, value in spike_features.items():
            try:
                sum_features.update({key: sum_features[key] + value})
            except KeyError:
                sum_features.update({key: value})
    avg_features = {key: value / len(cluster) for key, value in sum_features.items()}
    return avg_features
