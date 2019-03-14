from abc import ABCMeta, abstractmethod

from numpy import ndarray

NOT_IMPLEMENTED = "You should implement this."


class Loss:
    __metaclass__ = ABCMeta

    @abstractmethod
    def eval_data_loss(self, labels: ndarray, scores: ndarray):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def eval_gradient(self):
        raise NotImplementedError(NOT_IMPLEMENTED)
