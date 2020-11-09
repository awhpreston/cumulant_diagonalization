"""
This module faithfully implements Kernel Independent component analysis in Python
"""

import numpy as np


def amari_distance(matrix_1: np.ndarray, matrix_2: np.ndarray) -> float:
    """
    Get Amari distance between two non-singular matrices.
    Does not verify the axioms of a distance, always between 0 and 1.
    :param matrix_1: numpy array representing a matrix
    :param matrix_2: numpy array representing a second matrix
    :return: Amari distance, a value between 0 and 1
    """
    p_matrix = np.linalg.inv(matrix_1) @ matrix_2  # scaled permutation matrix
    p_abs = np.abs(p_matrix)

    max_rows = np.max(p_abs, axis=1)
    max_columns = np.max(p_abs, axis=0)

    p_row_max_normalized = p_abs / max_rows[:, None]
    p_column_max_normalized = p_abs / max_columns

    sum_max_normalized_rows = np.sum(p_row_max_normalized, axis=1)
    sum_max_normalized_columns = np.sum(p_column_max_normalized, axis=0)

    first_term = np.sum(sum_max_normalized_rows - 1)
    second_term = np.sum(sum_max_normalized_columns - 1)

    dimension = len(p_matrix)
    pre_factor = 1./(2*dimension*(dimension-1))
    return pre_factor*(first_term + second_term)
