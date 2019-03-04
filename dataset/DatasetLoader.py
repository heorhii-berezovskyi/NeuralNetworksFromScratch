from os.path import join

from numpy import genfromtxt


class DatasetLoader:
    def __init__(self, directory: str):
        self.directory = directory

    def load(self, train_dataset_name: str, test_dataset_name: str):
        train_dataset_path = join(self.directory, train_dataset_name)
        test_dataset_path = join(self.directory, test_dataset_name)

        train_dataset = genfromtxt(train_dataset_path, delimiter=',', skip_header=1, dtype=float)
        test_dataset = genfromtxt(test_dataset_path, delimiter=',', skip_header=1, dtype=float)

        return train_dataset, test_dataset
