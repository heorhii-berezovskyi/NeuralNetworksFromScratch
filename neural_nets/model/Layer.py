from abc import ABCMeta, abstractmethod

from numpy import ndarray

NOT_IMPLEMENTED = "You should implement this."


class Layer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, input_data: ndarray):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def backward(self, dout: ndarray):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def accept(self, visitor):
        raise NotImplementedError(NOT_IMPLEMENTED)


class LayerWithWeights(Layer):
    __metaclass__ = ABCMeta

    next_id = 0

    def __init__(self):
        self.id = LayerWithWeights.next_id
        LayerWithWeights.next_id += 1

    @abstractmethod
    def forward(self, input_data: ndarray):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def backward(self, dout: ndarray):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def get_weights(self):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def update_weights(self, w1: ndarray, w2: ndarray):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def get_gradients(self):
        raise NotImplementedError(NOT_IMPLEMENTED)

    def get_id(self):
        return self.id

    @abstractmethod
    def set_layer_mode(self, mode: str):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def accept(self, visitor):
        raise NotImplementedError(NOT_IMPLEMENTED)
