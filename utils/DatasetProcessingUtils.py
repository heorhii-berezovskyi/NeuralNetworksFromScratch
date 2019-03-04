from numpy import delete
from numpy import insert
from numpy import mean
from numpy import ndarray
from numpy.random import choice


def split_into_labels_and_data(dataset: ndarray):
    labels = dataset[:, 0]
    data = delete(dataset, 0, axis=1)
    return labels.astype(int), data


def preprocess_dataset(dataset: ndarray):
    dataset[:, 1:] -= mean(dataset[:, 1:])
    return dataset


def apply_bias_trick_on_dataset(dataset: ndarray):
    dataset = insert(dataset, 1, 1.0, axis=1)
    return dataset


def sample(dataset: ndarray, batch_size: int):
    return dataset[choice(dataset.shape[0], batch_size, replace=False), :]
