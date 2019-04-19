import os

import numpy as np


class DatasetLoader:
    def __init__(self, directory: str):
        self.directory = directory

    def load(self, train_dataset_name: str, test_dataset_name: str):
        train_dataset_path = os.path.join(self.directory, train_dataset_name)
        train_dataset = np.load(train_dataset_path)
        train_labels = train_dataset[:, 0].astype(np.uint8)
        train_data = np.delete(train_dataset, 0, axis=1).astype(dtype=np.float64)

        test_dataset_path = os.path.join(self.directory, test_dataset_name)
        test_dataset = np.load(test_dataset_path)
        test_labels = test_dataset[:, 0].astype(np.uint8)
        test_data = np.delete(test_dataset, 0, axis=1).astype(dtype=np.float64)
        return train_labels, train_data, test_labels, test_data
