from abc import ABCMeta, abstractmethod

from numpy import ndarray

from neural_nets.model.Cache import Cache

NOT_IMPLEMENTED = "You should implement this."


class Loss:
    __metaclass__ = ABCMeta

    @abstractmethod
    def eval_data_loss(self, labels: ndarray, model_forward_run: list) -> tuple:
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def eval_gradient(self, loss_run: Cache) -> ndarray:
        raise NotImplementedError(NOT_IMPLEMENTED)
