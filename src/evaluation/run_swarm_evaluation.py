"""
This is the script to run evaluations of algorithms on Swarm data
"""

import numpy as np
from sklearn.decomposition import FastICA
from algorithms.forward_deep_ica import DeepPowerICA, DeepPowerPCA
from data_handling.global_data_params import DEFAULT_LOCAL_PATH, SWARM_FLOCKING_RELATIVE_PATH
from data_handling.read_swarm_data import read_csv, split_swarm_data_into_features_and_labels
from evaluation.cumulant_viewers import get_coskew_arrays, get_cokurt_arrays, \
    plot_covariance_histograms, plot_skew_histograms, plot_kurtosis_histograms


if __name__ == "__main__":

    print("Reading swarm data")
    out_dict = read_csv(DEFAULT_LOCAL_PATH / SWARM_FLOCKING_RELATIVE_PATH)
    data, _ = split_swarm_data_into_features_and_labels(out_dict)
    print(f"Dimensions are {np.shape(data)}")  # expecting (24016, 2400)

    print("Extracting independent features")

    # fast_ica = FastICA(n_components=20)
    # transformed_data = fast_ica.fit_transform(data)

    # deep_power_pca = DeepPowerPCA(layers=[40, 30, 20])
    # transformed_data = deep_power_pca.fit_transform(data)

    deep_power_ica = DeepPowerICA(layers=[30, 25, 20])
    transformed_data = deep_power_ica.fit_transform(data)

    print("Showing covariance")
    plot_covariance_histograms(samples=transformed_data, diagonal_bins=5, off_diagonal_bins=10)
    print("Calculating coskewness")
    coskew_arrays = get_coskew_arrays(data=transformed_data, regularization=0.01)
    plot_skew_histograms(coskew_arrays=coskew_arrays, diagonal_bins=5, semi_diagonal_bins=10, off_diagonal_bins=50)
    print("Calcluating co-excess-kurtosis")
    cokurtosis_arrays = get_cokurt_arrays(data=transformed_data, regularization=0.01)
    plot_kurtosis_histograms(cokurtosis_arrays=cokurtosis_arrays, diagonal_bins=5, mostly_diagonal_bins=10,
                             semi_diagonal_bins=20, mostly_off_diagonal_bins=30, off_diagonal_bins=50)
    print("Done")