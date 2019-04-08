from abc import ABCMeta, abstractmethod

from neural_nets.model.BatchNorm1D import BatchNorm1DTrain
from neural_nets.model.BatchNorm2D import BatchNorm2DTrain
from neural_nets.model.Conv2D import Conv2DTrain
from neural_nets.model.Dropout1D import Dropout1DTrain
from neural_nets.model.Dropout2D import Dropout2DTrain
from neural_nets.model.Linear import LinearTrain
from neural_nets.model.MaxPool import MaxPoolTrain
from neural_nets.model.Relu import ReluTrain

NOT_IMPLEMENTED = "You should implement this."


class Visitor:
    """
    Visitor design pattern representative. Visits each type of train mode model layers.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def visit_linear(self, layer: LinearTrain):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_conv2d(self, layer: Conv2DTrain):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_relu(self, layer: ReluTrain):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_max_pool(self, layer: MaxPoolTrain):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_batch_norm_1d(self, layer: BatchNorm1DTrain):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_batch_norm_2d(self, layer: BatchNorm2DTrain):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_dropout1d(self, layer: Dropout1DTrain):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_dropout2d(self, layer: Dropout2DTrain):
        raise NotImplementedError(NOT_IMPLEMENTED)
