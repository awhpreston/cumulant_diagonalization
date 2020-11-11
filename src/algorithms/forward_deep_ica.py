"""
This module contains proposed forward-fit deep ICA functions.
"""

import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import PowerTransformer
from typing import List


class DeepPowerICA(object):
    """
    An object that fits the deep power transform ICA. Not only transform forwards, but also backwards.
    """
    def __init__(self, layers: List[int], tol: float = 0.00001, verbose_training: bool = True) -> None:
        """
        Initialization method.
        :param layers: component dimension at each step (one layer is traditional ICA), activation used in hidden layers
        :param tol: the tolerance for ICA as a float
        :param verbose_training: Boolean, True to have the fit method print training progress
        """
        self.layers = layers
        self.n_layers = len(layers)
        self.tol = tol
        self.ica_list = []
        self.power_list = []
        self.verbose_training = verbose_training
        for i, width in enumerate(layers):
            ica_object = FastICA(n_components=width, tol=tol)
            power_object = PowerTransformer(method='yeo-johnson')
            self.ica_list.append(ica_object)
            if i != len(layers)-1:
                self.power_list.append(power_object)
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        The method that fits the ICA transform function and returns the transformed result.
        :param data: the training data as a numpy array of shape [samples, input dimensions]
        :return: Transformed data as a numpy array with shape [samples, output dimensions]
        """
        for i in range(self.n_layers):
            if self.verbose_training:
                print("Fitting layer %d with output width %d" % (i+1, self.layers[i]))
            new_data = np.nan_to_num(data)
            new_data = self.ica_list[i].fit_transform(X=new_data)
            if i != self.n_layers - 1:
                self.power_list[i].fit(new_data)
                new_data = self.power_list[i].inverse_transform(new_data)
            data = new_data
        return data

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Method that transforms data using pre-trained objects.
        :param data: the training data as a numpy array of shape [samples, input dimensions]
        :return: Transformed data as a numpy array with shape [samples, output dimensions]
        """
        for i in range(self.n_layers):
            new_data = np.nan_to_num(data)
            new_data = self.ica_list[i].transform(X=new_data)
            if i != self.n_layers - 1:
                new_data = self.power_list[i].inverse_transform(new_data)
            data = new_data
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Method that inverse transforms data using pre-trained objects.
        :param data: the training data as a numpy array of shape [samples, output dimensions]
        :return: Transformed data as a numpy array with shape [samples, input dimensions]
        """
        for i in range(self.n_layers):
            j = self.n_layers-1-i
            new_data = np.nan_to_num(data)
            new_data = self.ica_list[j].inverse_transform(X=new_data)
            if i != self.n_layers - 1:
                new_data = self.power_list[j-1].transform(new_data)
            data = new_data
        return data


class DeepPowerPCA(object):
    """
    An object that fits the deep power transform using PCA instead of ICA. The motivation is both speed and keeping
    high-variance directions, while searching for non-Gaussianity and diagonal covariance.
    """
    def __init__(self, layers: List[int], tol: float = 0.00001, verbose_training: bool = True) -> None:
        """
        Initialization method.
        :param layers: component dimension at each step (one layer is traditional PCA), activation used in hidden layers
        :param tol: the tolerance for PCA as a float
        :param verbose_training: Boolean, True to have the fit method print training progress
        """
        self.layers = layers
        self.n_layers = len(layers)
        self.tol = tol
        self.pca_list = []
        self.power_list = []
        self.verbose_training = verbose_training
        for i, width in enumerate(layers):
            pca_object = PCA(n_components=width, tol=tol)
            power_object = PowerTransformer(method='yeo-johnson')
            self.pca_list.append(pca_object)
            if i != len(layers)-1:
                self.power_list.append(power_object)
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        The method that fits the ICA transform function and returns the transformed result.
        :param data: the training data as a numpy array of shape [samples, input dimensions]
        :return: Transformed data as a numpy array with shape [samples, output dimensions]
        """
        for i in range(self.n_layers):
            if self.verbose_training:
                print("Fitting layer %d with output width %d" % (i+1, self.layers[i]))
            new_data = np.nan_to_num(data)
            new_data = self.pca_list[i].fit_transform(X=new_data)
            if i != self.n_layers - 1:
                self.power_list[i].fit(new_data)
                new_data = self.power_list[i].inverse_transform(new_data)
            data = new_data
        return data

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Method that transforms data using pre-trained objects.
        :param data: the training data as a numpy array of shape [samples, input dimensions]
        :return: Transformed data as a numpy array with shape [samples, output dimensions]
        """
        for i in range(self.n_layers):
            new_data = np.nan_to_num(data)
            new_data = self.pca_list[i].transform(X=new_data)
            if i != self.n_layers - 1:
                new_data = self.power_list[i].inverse_transform(new_data)
            data = new_data
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Method that inverse transforms data using pre-trained objects.
        :param data: the training data as a numpy array of shape [samples, output dimensions]
        :return: Transformed data as a numpy array with shape [samples, input dimensions]
        """
        for i in range(self.n_layers):
            j = self.n_layers-1-i
            new_data = np.nan_to_num(data)
            new_data = self.pca_list[j].inverse_transform(X=new_data)
            if i != self.n_layers - 1:
                new_data = self.power_list[j-1].transform(new_data)
            data = new_data
        return data
