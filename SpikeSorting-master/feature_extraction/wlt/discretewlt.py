import matplotlib.pyplot as plt
import numpy as np
import pywt

from feature_extraction import derivatives as deriv
from scipy.stats import norm, kstest


#
# def new_ks(spikes):
#     for spike in spikes:
#         coeffsmatrix = haardecomposition(spike, 1)
#

def plot_dwt1(spike):
    coeffsmatrix = haardecomposition(spike, 1)
    ca, cd = coeffsmatrix
    time = np.arange(79)
    fig = plt.figure(figsize=(5, 6))
    axes = fig.subplots(3)
    axes[0].set_title("Spike signal")
    axes[0].plot(time, spike)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Magnitude")
    plt.tight_layout()
    axes[1].set_title("Approx coeff")
    axes[1].plot(np.arange(len(ca)), ca)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Magnitude")
    plt.tight_layout()
    axes[2].set_title("Detail coeff")
    axes[2].plot(np.arange(len(cd)), cd)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Magnitude")
    plt.tight_layout()
    plt.show()


def plot_dwt(spike):
    coeffsmatrix = haardecomposition(spike, 4)
    ca, cd4, cd3, cd2, cd1 = coeffsmatrix
    time = np.arange(79)
    fig = plt.figure(figsize=(5, 10))
    axes = fig.subplots(6)
    axes[0].set_title("Spike signal")
    axes[0].plot(time, spike)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Magnitude")
    plt.tight_layout()
    axes[1].set_title("Approx coeff")
    axes[1].plot(np.arange(len(ca)), ca)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Magnitude")
    plt.tight_layout()
    axes[2].set_title("Detail coeff4")
    axes[2].plot(np.arange(len(cd4)), cd4)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Magnitude")
    plt.tight_layout()
    axes[3].set_title("Detail coeff3")
    axes[3].plot(np.arange(len(cd3)), cd3)
    axes[3].set_xlabel("Time")
    axes[3].set_ylabel("Magnitude")
    plt.tight_layout()
    axes[4].set_title("Detail coeff2")
    axes[4].plot(np.arange(len(cd2)), cd2)
    axes[4].set_xlabel("Time")
    axes[4].set_ylabel("Magnitude")
    plt.tight_layout()
    axes[5].set_title("Detail coeff1")
    axes[5].plot(np.arange(len(cd1)), cd1)
    axes[5].set_xlabel("Time")
    axes[5].set_ylabel("Magnitude")
    plt.tight_layout()
    plt.show()


def dwt_pca(spikes):
    result = []
    for spike in spikes:
        coeffsmatrix = haardecomposition(spike, 4)
        ca, cd4, cd3, cd2, cd1 = coeffsmatrix
        res1 = []
        for i in np.arange(len(ca)):
            res1.append(ca[i])
        for i in np.arange(len(cd1)):
            res1.append(cd1[i])
        for i in np.arange(len(cd2)):
            res1.append(cd2[i])
        for i in np.arange(len(cd3)):
            res1.append(cd3[i])
        for i in np.arange(len(cd4)):
            res1.append(cd4[i])
        result.append(res1)
    return np.array(result)


def dwt_fd_method(spikes):
    result = []
    for spike in spikes:
        coeffsmatrix = haardecomposition(spike, 4)
        ca, cd4, cd3, cd2, cd1 = coeffsmatrix
        res1 = []
        if len(ca) > 4:
            res1.append(deriv.compute_fdmethod_1spike(ca))
        else:
            res1.append(deriv.compute_fdmethod_1spike2(ca))
        if len(cd4) > 4:
            res1.append(deriv.compute_fdmethod_1spike(cd4))
        else:
            res1.append(deriv.compute_fdmethod_1spike2(cd4))
        if len(cd3) > 4:
            res1.append(deriv.compute_fdmethod_1spike(cd3))
        else:
            res1.append(deriv.compute_fdmethod_1spike2(cd3))
        if len(cd2) > 4:
            res1.append(deriv.compute_fdmethod_1spike(cd2))
        else:
            res1.append(deriv.compute_fdmethod_1spike2(cd2))
        if len(cd1) > 4:
            res1.append(deriv.compute_fdmethod_1spike(cd1))
        else:
            res1.append(deriv.compute_fdmethod_1spike2(cd1))
        result.append(np.ndarray.flatten(np.array(res1)))
    return result


def compute_haar_ks(spikes):
    result = []
    # print(len(spikes))
    for spike in spikes:
        coeffsmatrix = haardecomposition(spike, 4)
        ca, cd4, cd3, cd2, cd1 = coeffsmatrix
        coeffs = np.concatenate((ca, cd4, cd3, cd2, cd1))
        coeffs = np.ndarray.flatten(coeffs)
        # print(len(coeffs)) = 80
        result.append(coeffs)
    # print(len(result))
    k_s = []
    for coeff in range(79):
        coeffs = []
        for spike_coeff_pos in range(0, len(result)):
            coeffs.append(result[spike_coeff_pos][coeff])
        loc, scale = norm.fit(coeffs)
        # create a normal distribution with loc and scale
        n = norm(loc=loc, scale=scale)
        s, p = kstest(coeffs, n.cdf)
        k_s.append(p)
    min_10_pos = take_min_10positions(k_s)
    res = []
    for res_index in range(len(result)):
        lst = []
        for i in range(len(min_10_pos)):
            lst.append(result[res_index][i])
        res.append(lst)
    return np.array(res)


def compute_haar(spikes):
    # without approx coeff
    # coeffsmatrix[0] - approximation coefficient
    # coeffsmatrix[1] - details coefficient

    result = []

    for spike in spikes:
        coeffsmatrix = haardecomposition(spike, 4)
        ca, cd4, cd3, cd2, cd1 = coeffsmatrix
        coeffs = np.concatenate((cd4, cd3, cd2, cd1))
        variance = np.var(coeffs)
        std = np.std(coeffs)
        mean = np.mean(coeffs)
        coeffs = coeffs[(coeffs >= mean - 3 * std) & (coeffs <= mean + 3 * std)]  # gets rid of outliers
        gauss = np.random.normal(mean, std, len(coeffs))
        cdf = np.sort(coeffs)
        cdfGauss = np.sort(gauss)
        lilliefors = np.zeros(len(cdf))
        for i in np.arange(0, len(cdf)):
            lilliefors[i] = (abs(np.argwhere(cdf == coeffs[i])[0] - np.argwhere(cdfGauss == gauss[i])[0]))
        respos = take_max_10positions(lilliefors)
        res1 = []
        respos = np.ndarray.flatten(respos)
        for i in range(0, len(respos)):
            res1.append(coeffs[respos[i]])
        result.append(res1)

    return result


def compute_haar2(spikes):
    # with approx coeff
    # coeffsmatrix[0] - approximation coefficient
    # coeffsmatrix[1] - details coefficient

    result = []

    for spike in spikes:
        coeffsmatrix = haardecomposition(spike, 4)
        ca, cd4, cd3, cd2, cd1 = coeffsmatrix
        coeffs = np.concatenate((ca, cd4, cd3, cd2, cd1))
        variance = np.var(coeffs)
        std = np.std(coeffs)
        mean = np.mean(coeffs)
        coeffs = coeffs[(coeffs >= mean - 3 * std) & (coeffs <= mean + 3 * std)]  # gets rid of outliers
        gauss = np.random.normal(mean, std, len(coeffs))
        cdf = np.sort(coeffs)
        cdfGauss = np.sort(gauss)
        lilliefors = np.zeros(len(cdf))
        for i in np.arange(0, len(cdf)):
            lilliefors[i] = (abs(np.where(cdf == coeffs[i])[0] - np.where(cdfGauss == gauss[i])[0]))
        respos = take_max_10positions(lilliefors)
        res1 = []
        respos = np.ndarray.flatten(respos)
        for i in range(0, len(respos)):
            res1.append(coeffs[respos[i]])
        result.append(res1)
    return np.array(result)


def haardecomposition(spike, level):
    coeffsmatrix = pywt.wavedec(spike, 'haar', mode='per', level=level)
    return coeffsmatrix


def take_min_10positions(coeff):
    ordered = coeff.copy()
    ordered.sort()
    result_pos = []
    for i in range(0, 10):
        result_pos.append(coeff.index(ordered[i]))
    return result_pos


def take_max_10positions(coeff):
    res = np.argsort(coeff)[-10:][::-1]
    return res


def returnCoefficients(spikes, lvl):
    result = []
    for spike in spikes:
        coeffsmatrix = haardecomposition(spike, lvl)
        coeffs = []
        for i in np.arange(0, lvl + 1):
            coeffs = np.concatenate((coeffs, coeffsmatrix[i]))
        result.append(coeffs)
    return result


def returnCoefficient(spikes, lvl):
    result = []
    for spike in spikes:
        coeffsmatrix = haardecomposition(spike, lvl)
        result.append(coeffsmatrix[2])
    return result


def testplots(spikes):
    coeffsmatrix = pywt.dwt(spikes[0], 'haar')
    coeffs = np.concatenate((coeffsmatrix[0], coeffsmatrix[1]))
    variance = np.var(coeffs)
    std = np.std(coeffs)
    mean = np.mean(coeffs)
    gauss = np.random.normal(mean, std, 80)
    count, bins, ignored = plt.hist(gauss, 50, density=True)
    plt.plot(bins, 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (bins - mean) ** 2 / (2 * std ** 2)), linewidth=2,
                      color='r')
    plt.show()

    plt.plot(np.arange(len(gauss)), gauss)
    plt.plot(np.arange(len(coeffs)), coeffs)
    plt.show()

    # for i in range(0, len(spikes[0]), 300):
    # plt.plot(np.arange(len(spikes[0])), spikes[0])  # blue
    # coeff0 = pywt.dwt(spikes[0], 'haar')[0]  # orange approx
    # plt.plot(np.arange(40), coeff0)
    # coeff0 = pywt.dwt(spikes[0], 'haar')[1]  # green details
    # plt.plot(np.arange(40), coeff0)
    # plt.show()


def plotspike0coeffs(spikes):
    coeffsmatrix = haardecomposition(spikes[0], 4)
    ca, cd4, cd3, cd2, cd1 = coeffsmatrix
    coeffs = np.concatenate((ca, cd4, cd3, cd2, cd1))
    plt.plot(np.arange(0, len(spikes[0])), spikes[0])
    plt.plot(np.arange(0, len(ca)), ca)
    plt.plot(np.arange(len(ca) - 1, len(cd4) + len(ca) - 1), cd4)
    plt.plot(np.arange(len(cd4) + len(ca) - 2, len(cd4) + len(ca) - 2 + len(cd3)), cd3)
    plt.plot(np.arange(len(cd4) + len(ca) - 3 + len(cd3), len(cd4) + len(ca) - 3 + len(cd3) + len(cd2)), cd2)
    plt.plot(np.arange(len(cd4) + len(ca) - 4 + len(cd3) + len(cd2),
                                len(cd4) + len(ca) - 4 + len(cd3) + len(cd2) + len(cd1)), cd1)

    plt.show()
