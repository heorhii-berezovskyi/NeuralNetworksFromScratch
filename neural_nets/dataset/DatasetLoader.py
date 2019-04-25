import numpy as np
from numpy import ndarray


class DatasetLoader:
    def load(self, path: str) -> (ndarray, ndarray):
        """
        Loads dataset from npy file and returns labels and data.
        :param dataset_name: name of a dataset file with extension.
        :return: tuple, containing labels and data.
        """
        dataset = np.load(path)
        labels = dataset[:, 0].astype(np.uint8)
        data = np.delete(dataset, 0, axis=1).astype(dtype=float)
        return labels, data
