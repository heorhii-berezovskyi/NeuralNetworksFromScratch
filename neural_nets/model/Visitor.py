from abc import ABCMeta, abstractmethod

from neural_nets.model.Layer import TrainModeLayer, TrainModeLayerWithWeights

NOT_IMPLEMENTED = "You should implement this."


class TrainLayerVisitor:
    """
    Visitor design pattern representative. Visits each type of train mode model layers.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def visit_affine_train(self, layer: TrainModeLayerWithWeights):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_batch_norm_1d_train(self, layer):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_batch_norm_2d_train(self, layer):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_weightless_train(self, layer: TrainModeLayer):
        raise NotImplementedError(NOT_IMPLEMENTED)
