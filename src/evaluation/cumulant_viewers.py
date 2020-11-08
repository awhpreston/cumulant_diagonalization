"""
This contains functions and classes for viewing (multivariate) cumulants
"""

from dataclasses import dataclass
from decimal import Decimal
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from typing import Dict


def compute_univariate_cumulants(data: np.ndarray, standardize: bool = False,
                                 regularization: float = 0.1) -> Dict[int, float]:
    """
    Calculates univariate cumulants of a dataset and returns a dictionary of cumulants by order, up to seven
    :param data: a numpy array structured as [sample][dimension]
    :param standardize: True to standardize cumulants beyond second order
    :param regularization: if standardizing, this adds a small constant to the standard deviations used
    :return: dictionary with order as key and cumulant as value
    """
    mean = np.mean(data, axis=0)

    data -= mean

    second_moment = np.mean(np.array(data) ** 2, axis=0)
    if standardize:
        data /= np.sqrt(second_moment) + regularization
    third_moment = np.mean(np.array(data) ** 3, axis=0)
    fourth_moment = np.mean(np.array(data) ** 4, axis=0)
    fifth_moment = np.mean(np.array(data) ** 5, axis=0)
    sixth_moment = np.mean(np.array(data) ** 6, axis=0)
    seventh_moment = np.mean(np.array(data) ** 7, axis=0)

    skew = third_moment
    kurt = fourth_moment - 3 * second_moment ** 2
    fifth_cumulant = fifth_moment - 10 * third_moment * second_moment
    sixth_cumulant = sixth_moment - 5 * fourth_moment - 10 * skew ** 2 - 10 * kurt
    seventh_cumulant = seventh_moment - 6*fifth_moment - 15*skew*fourth_moment - 20*kurt*skew - 15*fifth_cumulant

    cumulant_dict = dict()
    cumulant_dict[1] = mean
    cumulant_dict[2] = second_moment
    cumulant_dict[3] = skew
    cumulant_dict[4] = kurt
    cumulant_dict[5] = fifth_cumulant
    cumulant_dict[6] = sixth_cumulant
    cumulant_dict[7] = seventh_cumulant
    return cumulant_dict


def coskew_maps(samples: np.ndarray, standardize: bool = True) -> np.ndarray:
    """
    Creates a 2D array of the coskews as a flattened 3D coskew array.
    :param samples: samples structured as [sample][dimension]
    :param standardize: True to standardize the samples
    :return: a 2D array, which is a partially flattened from of the coskew array [dimension, dimensions**2]
    """
    samples -= np.mean(samples, axis=0)
    if standardize:
        samples /= np.std(samples, axis=0)
    dimensions = len(samples[0])
    big3d = np.zeros([dimensions, dimensions, dimensions])
    for i in range(dimensions):
        for j in range(i, dimensions):
            for k in range(j, dimensions):
                score = np.mean(samples[:, i]*samples[:, j]*samples[:, k])
                big3d[i, j, k] = score
                big3d[i, k, j] = score
                big3d[j, k, i] = score
                big3d[k, j, i] = score
                big3d[j, i, k] = score
                big3d[k, i, j] = score
    big2d = np.reshape(big3d, [dimensions, dimensions**2])
    return big2d


@dataclass
class CoskewArrays:
    """
    Object to store measured coskew data.
    """
    diagonal: np.ndarray
    semi_diagonal: np.ndarray
    off_diagonal: np.ndarray


@dataclass
class CokurtosisArrays:
    """
    Object to store measured cokurtosis data.
    """
    diagonal: np.ndarray
    mostly_diagonal: np.ndarray
    semi_diagonal: np.ndarray
    mostly_off_diagonal: np.ndarray
    off_diagonal: np.ndarray


def get_coskew_arrays(data: np.ndarray, standardize: bool = True, regularization: float = 0.01) -> CoskewArrays:
    """
    Measures coskews and returns a CoskewArrays object. Contains diagonal coskews (i.e. skews), semi-diagonal coskews,
    e.g. "1,1,2", and one of completely off-diagonal coskews (e.g. "1,2,3"). These are
    unstructured, upper "triangle" coskews intended for use in histograms.
    *** scales as dimensions^3, reduce dimension to something sensible for most efficient results ***
    :param data: an array as [sample][dimension]
    :param standardize: whether or not to standardize (centralizing always happens)
    :param regularization: standard deviation shift away from zero, if standardizing
    :return: CoskewArrays object
    """
    diag_list = list()
    two_diag_list = list()
    off_diag_list = list()
    dimension = len(data[0])
    data -= np.mean(data, axis=0)
    if standardize:
        data /= np.std(data, axis=0) + regularization
    for i in range(dimension):
        for j in range(i, dimension):
            for k in range(j, dimension):
                skew = np.mean(data[:, i]*data[:, j]*data[:, k])
                if i == j == k:
                    diag_list.append(skew)
                elif i == j or i == k or j == k:
                    two_diag_list.append(skew)
                else:
                    off_diag_list.append(skew)
    diag_list = np.array(diag_list)
    two_diag_list = np.array(two_diag_list)
    off_diag_list = np.array(off_diag_list)
    return CoskewArrays(diagonal=diag_list, semi_diagonal=two_diag_list, off_diagonal=off_diag_list)


def get_cokurt_arrays(data, standardize=True, regularization=0.01) -> CokurtosisArrays:
    """
    Measures co-excess-kurtoses to return a CokurtosisArrays object. Contains diagonal cokurtoses,
    mostly diagonal cokurtoses, e.g. "1,1,1,2", semi-diagonal cokurts, e.g. "1,1,2,2",
    mostly-off-diagonal, e.g. "1,1,2,3", and completely off-diagonal cokurts (e.g. "1,2,3,4"). These are
    unstructured, upper "triangle" cokurts intended for use in histograms.
    *** scales as dimensions^4, reduce dimension to something sensible for most efficient results ***
    :param data: an array as [sample][dimension]
    :param standardize: whether or not to standardize (centralizing always happens)
    :param regularization: standard deviation shift away from zero, if standardizing
    :return: CokurtosisArrays object
    """
    diag_list = list()
    three_diag_list = list()
    two_two_diag_list = list()
    two_diag_list = list()
    off_diag_list = list()
    dimension = len(data[0])
    data -= np.mean(data, axis=0)
    if standardize:
        data /= np.std(data, axis=0) + regularization
    for i in range(dimension):
        for j in range(i, dimension):
            for k in range(j, dimension):
                for l in range(k, dimension):
                    kurt = np.mean(data[:, i]*data[:, j]*data[:, k]*data[:, l])
                    kurt -= np.mean(data[:, i]*data[:, j])*np.mean(data[:, k]*data[:, l])
                    kurt -= np.mean(data[:, i]*data[:, k])*np.mean(data[:, j]*data[:, l])
                    kurt -= np.mean(data[:, i]*data[:, l])*np.mean(data[:, k]*data[:, j])

                    counter = Counter([i, j, k, l]).values()
                    if 4 in counter:
                        diag_list.append(kurt)
                    elif 3 in counter:
                        three_diag_list.append(kurt)
                    elif 2 not in counter:
                        off_diag_list.append(kurt)
                    elif 1 not in counter:
                        two_two_diag_list.append(kurt)
                    else:
                        two_diag_list.append(kurt)

    diag_list = np.array(diag_list)
    two_diag_list = np.array(two_diag_list)
    two_two_diag_list = np.array(two_two_diag_list)
    three_diag_list = np.array(three_diag_list)
    off_diag_list = np.array(off_diag_list)
    return CokurtosisArrays(diagonal=diag_list, mostly_diagonal=three_diag_list, semi_diagonal=two_two_diag_list,
                            mostly_off_diagonal=two_diag_list, off_diagonal=off_diag_list)


def plot_covariance_histograms(samples: np.ndarray, diagonal_bins: int = 5, off_diagonal_bins: int = 10,
                               standardize: bool = True) -> None:
    """
    Plots histograms showing the covariance distributions taking the samples as input.
    :param samples: a numpy array shaped as [sample number][dimension number]
    :param diagonal_bins: number of bins for the histogram of diagonal covariance values (i.e. variances)
    :param off_diagonal_bins: number of bins for the histogram of off-diagonal covariance values
    :param standardize: whether or not to standardize (centralizing always happens)
    :return: technically None, but plots a histogram figure with diagonals, nearly diagonals, and off-diagonals
    """
    cov = np.cov(samples, rowvar=False)
    stds = np.std(samples, axis=0)
    dimensions = len(samples[0])
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    deviations = np.abs(cov)
    diag_devs = []
    diag_raw = []
    for i in range(dimensions):
        diag_devs.append(deviations[i, i])
        diag_raw.append(cov[i, i])
    diag_devs = np.array(diag_devs)
    median = float(np.median(diag_devs))
    mean = float(np.mean(diag_devs))
    plt.hist(diag_raw, bins=diagonal_bins)
    plt.title('Mean: %.2E, \n median: %.2E' % (Decimal(mean), Decimal(median)))
    fig.add_subplot(1, 2, 2)
    off_diag_devs = []
    off_diag_raw = []
    for i in range(dimensions):
        for j in range(i+1, dimensions):
            abs_score = deviations[i, j]
            raw_score = cov[i, j]
            if standardize:
                abs_score /= stds[i]
                abs_score /= stds[j]
                raw_score /= stds[i]
                raw_score /= stds[j]
            off_diag_devs.append(abs_score)
            off_diag_raw.append(raw_score)
    off_diag_devs = np.array(off_diag_devs)
    median = float(np.median(off_diag_devs))
    mean = float(np.mean(off_diag_devs))
    plt.hist(off_diag_raw, bins=off_diagonal_bins)
    plt.title('Mean: %.2E, \n median: %.2E' % (Decimal(mean), Decimal(median)))
    plt.show()


def plot_skew_histograms(coskew_arrays: CoskewArrays, diagonal_bins: int = 5, semi_diagonal_bins: int = 10,
                         off_diagonal_bins: int = 50) -> None:
    """
    Plots histograms showing the coskew distributions taking the output of coskew_lists(:) as input
    :param coskew_arrays: a CoskewArrays object
    :param diagonal_bins: number of bins for diagonal coskew values (i.e. skews)
    :param semi_diagonal_bins: bins for semi-diagonal coskew values
    :param off_diagonal_bins: bins for fully off-diagonal values
    """
    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    deviation = np.abs(coskew_arrays.diagonal)
    median = float(np.median(deviation))
    mean = float(np.mean(deviation))
    plt.hist(coskew_arrays.diagonal, bins=diagonal_bins)
    plt.title('Mean: %.2E, \n median: %.2E' % (Decimal(mean), Decimal(median)))
    fig.add_subplot(1, 3, 2)
    deviation = np.abs(coskew_arrays.semi_diagonal)
    median = float(np.median(deviation))
    mean = float(np.mean(deviation))
    plt.hist(coskew_arrays.semi_diagonal, bins=semi_diagonal_bins)
    plt.title('Mean: %.2E, \n median: %.2E' % (Decimal(mean), Decimal(median)))
    fig.add_subplot(1, 3, 3)
    deviation = np.abs(coskew_arrays.off_diagonal)
    median = float(np.median(deviation))
    mean = float(np.mean(deviation))
    plt.hist(coskew_arrays.off_diagonal, bins=off_diagonal_bins)
    plt.title('Mean: %.2E, \n median: %.2E' % (Decimal(mean), Decimal(median)))
    plt.show()


def plot_kurtosis_histograms(cokurtosis_arrays: CokurtosisArrays,
                             diagonal_bins: int = 5,
                             mostly_diagonal_bins: int = 10,
                             semi_diagonal_bins: int = 20,
                             mostly_off_diagonal_bins: int = 30,
                             off_diagonal_bins: int = 50):
    """
    Plots histograms showing the cokurt distributions taking the output of cokurt_lists(:) as input
    :param cokurtosis_arrays: CokurtosisArrays object
    :param diagonal_bins: bins for diagonal co-kurtosis values (i.e. excess kurtoses)
    :param mostly_diagonal_bins: bins for nearly-diagonal cokurtosis entries x-x-x-y
    :param semi_diagonal_bins: bins for semi-diagonal cokurtosis entries x-x-y-y
    :param mostly_off_diagonal_bins: bins for nearly-off-diagonal entries x-x-y-z
    :param off_diagonal_bins: bins for fully-off-diagonal entries x-y-z-t
    """
    fig = plt.figure()
    fig.add_subplot(1, 5, 1)
    deviation = np.abs(cokurtosis_arrays.diagonal)
    median = float(np.median(deviation))
    mean = float(np.mean(deviation))
    plt.hist(cokurtosis_arrays.diagonal, bins=diagonal_bins)
    plt.title('Mean: %.2E, \n median: %.2E' % (Decimal(mean), Decimal(median)))
    fig.add_subplot(1, 5, 2)
    deviation = np.abs(cokurtosis_arrays.mostly_diagonal)
    median = float(np.median(deviation))
    mean = float(np.mean(deviation))
    plt.hist(cokurtosis_arrays.mostly_diagonal, bins=mostly_diagonal_bins)
    plt.title('Mean: %.2E, \n median: %.2E' % (Decimal(mean), Decimal(median)))
    fig.add_subplot(1, 5, 3)
    deviation = np.abs(cokurtosis_arrays.semi_diagonal)
    median = float(np.median(deviation))
    mean = float(np.mean(deviation))
    plt.hist(cokurtosis_arrays.semi_diagonal, bins=semi_diagonal_bins)
    plt.title('Mean: %.2E, \n median: %.2E' % (Decimal(mean), Decimal(median)))
    fig.add_subplot(1, 5, 4)
    deviation = np.abs(cokurtosis_arrays.mostly_off_diagonal)
    median = float(np.median(deviation))
    mean = float(np.mean(deviation))
    plt.hist(cokurtosis_arrays.mostly_off_diagonal, bins=mostly_off_diagonal_bins)
    plt.title('Mean: %.2E, \n median: %.2E' % (Decimal(mean), Decimal(median)))
    fig.add_subplot(1, 5, 5)
    deviation = np.abs(cokurtosis_arrays.off_diagonal)
    median = float(np.median(deviation))
    mean = float(np.mean(deviation))
    plt.hist(cokurtosis_arrays.off_diagonal, bins=off_diagonal_bins)
    plt.title('Mean: %.2E, \n median: %.2E' % (Decimal(mean), Decimal(median)))
    plt.show()
    return
