from abc import ABCMeta, abstractmethod

from neural_nets.model.Layer import LayerWithWeights, Layer

NOT_IMPLEMENTED = "You should implement this."


class Visitor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def visit_layer(self, layer: Layer):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_layer_with_weights(self, layer: LayerWithWeights):
        raise NotImplementedError(NOT_IMPLEMENTED)


class WeightedLayersVisitor(Visitor):
    def __init__(self):
        self.layers_with_weights = []

    def reset(self):
        self.layers_with_weights.clear()

    def visit_layer(self, layer: Layer):
        pass

    def visit_layer_with_weights(self, layer: LayerWithWeights):
        self.layers_with_weights.append(layer)

    def get_layers_with_weights(self):
        return self.layers_with_weights
