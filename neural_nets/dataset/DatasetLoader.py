from os.path import join

import numpy as np


class DatasetLoader:
    def __init__(self, directory: str):
        self.directory = directory

    def load(self, train_dataset_name: str, test_dataset_name: str):
        train_dataset_path = join(self.directory, train_dataset_name)
        test_dataset_path = join(self.directory, test_dataset_name)

        # train_dataset = np.genfromtxt(train_dataset_path, delimiter=',', skip_header=1, dtype=float)
        # test_dataset = np.genfromtxt(test_dataset_path, delimiter=',', skip_header=1, dtype=float)
        train_dataset = np.load(train_dataset_path)
        test_dataset = np.load(test_dataset_path)
        return train_dataset.astype(dtype=np.float64), test_dataset.astype(dtype=np.float64)
