import numpy as np
from os.path import join


class DatasetFormatter:
    def to_npy(self, folder_path: str, file_name: str, new_name: str):
        path = join(folder_path, file_name)
        dataset = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=np.uint8)
        new_path = join(folder_path, new_name)
        np.save(new_path, dataset)
