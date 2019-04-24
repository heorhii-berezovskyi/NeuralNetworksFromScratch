import os

import numpy as np
from numpy import ndarray


class DatasetLoader:
    def __init__(self, directory=''):
        self.directory = directory

    def load(self, dataset_name: str) -> (ndarray, ndarray):
        """
        Loads dataset from npy file and returns labels and data.
        :param dataset_name: name of a dataset file with extension.
        :return: tuple, containing labels and data.
        """
        dataset_path = os.path.join(self.directory, dataset_name)
        dataset = np.load(dataset_path)
        labels = dataset[:, 0].astype(np.uint8)
        data = np.delete(dataset, 0, axis=1).astype(dtype=float)
        return labels, data
