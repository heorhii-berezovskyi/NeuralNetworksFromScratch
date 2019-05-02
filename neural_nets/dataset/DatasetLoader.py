import os

import numpy as np
from numpy import ndarray


class DatasetLoader:
    def __init__(self, dataset_directory: str):
        self.dir = dataset_directory

    def load(self, type: str) -> (ndarray, ndarray):
        data_path = os.path.join(self.dir, type + '_data.npy')
        data = np.load(data_path)

        labels_path = os.path.join(self.dir, type + '_labels.npy')
        labels = np.load(labels_path)
        return labels.astype(np.uint8), data.astype(float)
