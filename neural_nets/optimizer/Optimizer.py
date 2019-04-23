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
        Performs optimizer memory update based on the model backward run.
        :param layer_backward_run: is a data structure containing gradients by layer's weights.
        :return: updated optimizer with updated memory.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def update_weights(self, weights: Cache) -> Cache:
        """
        Performs weights update.
        :param weights: weights to update.
        :return: updated weights.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def memory_content(self) -> dict:
        """
        :return: dict containing optimizer state.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def from_params(self, all_params):
        """
        Loads optimizer memory from saved params.
        :param all_params: an opened .npy file.
        :return: optimizer instance with loaded memory state.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @classmethod
    @abstractmethod
    def init_memory(cls, layer_id: str, weights: Cache):
        """
        Performs memory initialization for specified weights and layer id.
        :param layer_id: is a layer identifier.
        :param weights: layer wights.
        :return: initialized optimizer instance.
        """
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
