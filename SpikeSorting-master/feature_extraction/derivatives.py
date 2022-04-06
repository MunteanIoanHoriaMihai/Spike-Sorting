import numpy as np
from scipy.ndimage import gaussian_filter1d


def compute_fdmethod_1spike(spike):
    first_derivative = compute_derivative5stencil(spike)
    f_min = min(first_derivative)
    f_max = max(first_derivative)
    result = []
    result.append(f_max - f_min)
    result.append(max(spike))
    return np.array(result)


def compute_fdmethod(spikes):
    """
    2D Dimensionality reduction method using the range of the first derivative, and the peak of the spikes as dimensions
    :param spikes: matrix - the list of spikes in a simulation
    """
    final_result = []

    for x in spikes:
        first_derivative = compute_derivative5stencil(x)
        f_min = min(first_derivative)
        f_max = max(first_derivative)
        result = []
        result.append(f_max - f_min)
        result.append(max(x))
        final_result.append(result)

    return np.array(final_result)


def compute_fdmethod3d(spikes):
    """
    3D Dimensionality reduction method using the max of the first derivative, min of the first derivative,
      and the peak of the spikes as dimensions
    :param spikes: matrix - the list of spikes in a simulation
    """
    final_result = []

    for x in spikes:
        first_derivative = compute_derivative5stencil(x)
        f_min_pos = compute_min_pos(first_derivative)
        f_max_pos = compute_max_pos(first_derivative)

        f_max = max(first_derivative)
        f_min = min(first_derivative)

        result = []
        result.append(f_max)
        result.append(f_min)
        result.append(max(x))
        final_result.append(result)

    return np.array(final_result)


def compute_first_second_derivative3d(spikes):
    """
    3D Dimensionality reduction method using the max of the first derivative, and max and min of the second derivative as dimensions
    :param spikes: matrix - the list of spikes in a simulation
    """
    final_result = []

    for x in spikes:
        first_derivative = compute_derivative5stencil(x)

        f_max_pos = compute_max_pos(first_derivative)

        second_derivative = compute_derivative5stencil(first_derivative)

        s_min_pos = compute_min_pos(second_derivative)
        s_max_pos = compute_max_pos(second_derivative)

        result = []
        result.append(x[f_max_pos])
        result.append(x[s_max_pos])
        result.append(x[s_min_pos])
        final_result.append(result)

    return np.array(final_result)


def compute_first_second_derivative(spikes):
    """
    2D Dimensionality reduction method using the average of min&max of the first derivative,
     and average of min&max of the second as dimensions
    :param spikes: matrix - the list of spikes in a simulation
    """
    final_result = []

    for x in spikes:
        first_derivative = compute_derivative5stencil(x)

        # f_min_pos = compute_min_pos(first_derivative)
        # f_max_pos = compute_max_pos(first_derivative)
        fmin = min(first_derivative)
        fmax = max(first_derivative)

        second_derivative = compute_derivative5stencil(first_derivative)

        # s_min_pos = compute_min_pos(second_derivative)
        # s_max_pos = compute_max_pos(second_derivative)
        smin = min(second_derivative)
        smax = max(second_derivative)

        result = []
        # f_pos, s_pos = method6(x[f_min_pos], x[f_max_pos], x[s_min_pos], x[s_max_pos])
        # f_pos, s_pos = method6(fmin, fmax,smin,smax)

        result.append(fmin)
        result.append(fmax)
        result.append(smax)
        result.append(smin)
        # result.append(f_pos)
        # result.append(s_pos)
        final_result.append(result)

    return np.array(final_result)


def fsde_method(spikes):
    final_result = []
    for x in spikes:
        first_derivative = compute_derivative(x)

        fmin = min(first_derivative)
        fmax = max(first_derivative)

        second_derivative = compute_derivative(first_derivative)

        smin = min(second_derivative)
        smax = max(second_derivative)

        result = []

        result.append(fmin)
        # result.append(fmax)
        # result.append(smax)
        result.append(fmax)
        result.append(smax)
        final_result.append(np.array(result))

    return np.array(final_result)


def method6(f_min, f_max, s_min, s_max):
    """
    Method for computing the 2 dimensions according to fsde dimensionality reduction method
    :param f_min: min value of the first derivative
    :param f_max: max value of the first derivative
    :param s_min: min value of the second derivative
    :param s_max: max value of the second derivative
    """
    first = (f_min + f_max) / 2
    second = (s_min + s_max) / 2
    return first, second


def method5(f_min, f_max, s_min, s_max):
    """
    Method for computing the 2 dimensions according to fsde dimensionality reduction method
    :param f_min: min value of the first derivative
    :param f_max: max value of the first derivative
    :param s_min: min value of the second derivative
    :param s_max: max value of the second derivative
    """
    f_pos = abs(f_max - f_min)
    s_pos = abs(s_max - s_min)
    return f_pos, s_pos


def gaussian_filter(spikes):
    return gaussian_filter1d(spikes, sigma=3, order=0, truncate=9.0)


def compute_min_pos(array):
    return array.index(min(array))


def compute_max_pos(array):
    return array.index(max(array))


def compute_derivative(function):
    """
    Computes the derivative of a function
    :param function: vector of values representing the function
    """
    first_derivative = []

    for i in range(1, len(function)):
        first_derivative.append(function[i] - function[i - 1])
    first_derivative.append(0)
    return first_derivative


def compute_derivative5stencil(function):
    """
    Computes the derivative of a function using the 5 point stencil method
    :param function: vector of values representing the function
    """
    first_derivative = []
    i = 2
    x = (-function[i + 2] + 8 * function[i + 1] - 8 * function[i - 1] + function[i - 2]) / 12
    first_derivative.append(x)
    first_derivative.append(x)
    for i in range(2, len(function) - 3):
        x = (-function[i + 2] + 8 * function[i + 1] - 8 * function[i - 1] + function[i - 2]) / 12
        first_derivative.append(x)

    i = len(function) - 3
    x = (-function[i + 2] + 8 * function[i + 1] - 8 * function[i - 1] + function[i - 2]) / 12

    first_derivative.append(x)
    first_derivative.append(x)
    first_derivative.append(x)

    return first_derivative


def compute_fdmethod_1spike(spike):
    first_derivative = compute_derivative5stencil(spike)
    f_min = min(first_derivative)
    f_max = max(first_derivative)
    result = []
    result.append(f_max - f_min)
    result.append(max(spike))
    return np.array(result)


def compute_fdmethod_1spike2(spike):
    first_derivative = compute_derivative(spike)
    f_min = min(first_derivative)
    f_max = max(first_derivative)
    result = []
    result.append(f_max - f_min)
    result.append(max(spike))
    return np.array(result)
