"""
This is the script to run evaluations of algorithms on DOW-Jones data
"""

from sklearn.decomposition import FastICA
from algorithms.forward_deep_ica import DeepPowerICA, DeepPowerPCA
from data_handling.read_dow_jones import read_csv, from_dict_to_dow_jones_entries, from_dow_jones_entries_to_numpy
from evaluation.cumulant_viewers import get_coskew_arrays, get_cokurt_arrays, \
    plot_covariance_histograms, plot_skew_histograms, plot_kurtosis_histograms


if __name__ == "__main__":
    data = from_dow_jones_entries_to_numpy(from_dict_to_dow_jones_entries(read_csv()))

    # fast_ica = FastICA(n_components=10)
    # transformed_data = fast_ica.fit_transform(data)
    #
    # deep_power_pca = DeepPowerPCA(layers=[12, 11, 10])
    # transformed_data = deep_power_pca.fit_transform(data)

    deep_power_ica = DeepPowerICA(layers=[12, 11, 10])
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
