"""
Module for loading and handling Swarm Behaviour data.
"""

import numpy as np
from data_handling.global_data_params import DEFAULT_LOCAL_PATH, \
    SWARM_ALIGNED_RELATIVE_PATH, SWARM_FLOCKING_RELATIVE_PATH, SWARM_GROUPED_RELATIVE_PATH
from data_handling.data_utils import read_csv
from typing import Dict, List, Tuple


def split_swarm_data_into_features_and_labels(swarm_dictionary: Dict[str, List]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Take in a Python dictionary of swarm data and place the input data into a numpy array (N, D) where N is the number
    of samples and D is the dimension of the data, and another numpy array of binary label values of shape (N).
    These are returned as a tuple in that order. The labels are assumed to be under the 'Class' key.
    :param swarm_dictionary: raw input dictionary, keys are descriptive strings, values are lists of values
    :return: (features as a numpy array [N, D], labels as a numpy array [N])
    """
    class_keys = ['Class', 'Class ']
    features = np.transpose(np.array([swarm_dictionary[key] for key in swarm_dictionary.keys()
                                      if key not in class_keys]))
    features[np.isin(features, ['', ' '])] = 0
    features = np.array(features, dtype=float)
    try:
        labels = np.array(swarm_dictionary['Class'], dtype=int)
    except KeyError:
        labels = np.array(swarm_dictionary['Class '], dtype=int)
    return features, labels


if __name__ == "__main__":
    out_dict = read_csv(DEFAULT_LOCAL_PATH / SWARM_FLOCKING_RELATIVE_PATH)
    fea, lab = split_swarm_data_into_features_and_labels(out_dict)
    print(np.shape(fea), fea.dtype)
    print(np.shape(lab), lab.dtype)
