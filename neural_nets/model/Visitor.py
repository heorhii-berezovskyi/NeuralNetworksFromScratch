from abc import ABCMeta, abstractmethod

from neural_nets.model.Layer import TrainModeLayer, TrainModeLayerWithWeights, TestModeLayer, TestModeLayerWithWeights

NOT_IMPLEMENTED = "You should implement this."


class TrainLayerBaseVisitor:
    """
    Visitor design pattern representative. Visits each type of train mode model layers.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def visit_affine_train(self, layer: TrainModeLayerWithWeights):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_batch_norm_train(self, layer: TrainModeLayerWithWeights):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_weightless_train(self, layer: TrainModeLayer):
        raise NotImplementedError(NOT_IMPLEMENTED)


class TestLayerBaseVisitor:
    """
    Visitor design pattern representative. Visits each type of train mode model layers.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def visit_affine_test(self, layer: TestModeLayerWithWeights):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_batch_norm_test(self, layer: TestModeLayerWithWeights):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_weightless_test(self, layer: TestModeLayer):
        raise NotImplementedError(NOT_IMPLEMENTED)
