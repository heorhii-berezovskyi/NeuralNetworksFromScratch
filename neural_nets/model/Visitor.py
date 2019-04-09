from abc import ABCMeta, abstractmethod

from neural_nets.model.Layer import TrainModeLayer, TrainModeLayerWithWeights

NOT_IMPLEMENTED = "You should implement this."


class Visitor:
    """
    Visitor design pattern representative. Visits each type of train mode model layers.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def visit_linear(self, layer: TrainModeLayerWithWeights):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_batch_norm(self, layer: TrainModeLayerWithWeights):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_weightless_layer(self, layer: TrainModeLayer):
        raise NotImplementedError(NOT_IMPLEMENTED)
