from abc import ABCMeta, abstractmethod

from numpy import ndarray

from neural_nets.model.Cache import Cache

NOT_IMPLEMENTED = "You should implement this."


class TestModeLayer:
    """
    Test mode layer representative.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, input_data: ndarray) -> ndarray:
        """
        Performs a forward pass of a layer based on the input data.
        :param input_data: is an input data of a layer.
        :return: output data of a layer.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)


class TrainModeLayer:
    """
    Representative of a trainable layer.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, input_data: ndarray, layer_forward_run: Cache) -> Cache:
        """
        Performs a forward pass of a layer, updating a test model parameters if required.
        :param input_data: is a layer input data.
        :param layer_forward_run: previous cache of this layer.
        :return: new layer forward run.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def backward(self, dout: ndarray, layer_forward_run: Cache) -> Cache:
        """
        Performs a backward pass of a layer, based on the layer forward run and the gradient by this layer.
        :param dout: is a gradient by this layer.
        :param layer_forward_run: is a layer forward run parameters.
        :return: gradient by input data and weights, if present.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def to_test(self, layer_forward_run: Cache) -> TestModeLayer:
        """
        Creates a test mode layer representative based on it's weights and test model parameters.
        :param layer_forward_run: is a Cache of a parameters, required to build a test model layers.
        :return: a test mode layer object.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def accept(self, visitor):
        raise NotImplementedError(NOT_IMPLEMENTED)


class TrainModeLayerWithWeights(TrainModeLayer):
    """
    Representative of a trainable layer with weights.
    """
    __metaclass__ = ABCMeta

    def __init__(self, block_name: str):
        self.block_name = block_name

    @abstractmethod
    def optimize(self, layer_backward_run: Cache):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def content(self, layer_forward_run: Cache) -> dict:
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def from_params(self, all_params) -> tuple:
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def with_optimizer(self, optimizer_class):
        raise NotImplementedError(NOT_IMPLEMENTED)
