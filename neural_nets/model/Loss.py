from abc import ABCMeta, abstractmethod
from neural_nets.model.Params import Params

from numpy import ndarray

NOT_IMPLEMENTED = "You should implement this."


class Loss:
    __metaclass__ = ABCMeta

    @abstractmethod
    def eval_data_loss(self, labels: ndarray, model_forward_run: list):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def eval_gradient(self, loss_run: Params):
        raise NotImplementedError(NOT_IMPLEMENTED)
