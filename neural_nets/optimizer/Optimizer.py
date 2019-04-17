from abc import ABCMeta, abstractmethod

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayerWithWeights, TrainModeLayer
from neural_nets.model.Visitor import TrainLayerVisitor

NOT_IMPLEMENTED = "You should implement this."


class Optimizer:
    """
    Optimizer representative. Optimizer updates train model weights.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def update_memory(self, layer_backward_run: Cache):
        """
        Performs train model weights update based on the model backward run.
        :param layer_backward_run: is a data structure containing gradients by layer's weights.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def update_weights(self, weights: Cache) -> Cache:
        raise NotImplementedError(NOT_IMPLEMENTED)


class WeightsUpdateVisitor(TrainLayerVisitor):
    def __init__(self, model_backward_run: list):
        self.model_backward_run = model_backward_run
        self.result = []

    def visit_affine_train(self, layer: TrainModeLayerWithWeights):
        layer_backward_run = self.model_backward_run.pop()
        new_layer = layer.optimize(layer_backward_run=layer_backward_run)
        self.result.append(new_layer)

    def visit_weightless_train(self, layer: TrainModeLayer):
        self.model_backward_run.pop()
        self.result.append(layer)

    def visit_batch_norm_1d_train(self, layer):
        layer_backward_run = self.model_backward_run.pop()
        new_layer = layer.optimize(layer_backward_run=layer_backward_run)
        self.result.append(new_layer)

    def visit_batch_norm_2d_train(self, layer):
        layer_backward_run = self.model_backward_run.pop()
        new_layer = layer.optimize(layer_backward_run=layer_backward_run)
        self.result.append(new_layer)

    def get_result(self) -> list:
        self.result.reverse()
        return self.result
