"""
This is the script to run evaluations of algorithms on DOW-Jones data
"""

from sklearn.decomposition import FastICA
from data_handling.read_dow_jones import read_csv, from_dict_to_dow_jones_entries, from_dow_jones_entries_to_numpy

if __name__ == "__main__":
    data = from_dow_jones_entries_to_numpy(from_dict_to_dow_jones_entries(read_csv()))
    fast_ica = FastICA(n_components=3)
    fast_ica.fit(data)
    print(fast_ica.components_)
