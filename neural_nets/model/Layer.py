from abc import ABCMeta, abstractmethod

from numpy import ndarray

from neural_nets.model.Cache import Cache

NOT_IMPLEMENTED = "You should implement this."


class TrainModeLayer:
    """
    Representative of a trainable layer.
    """
    __metaclass__ = ABCMeta

    next_id = 0

    def __init__(self):
        self.id = TrainModeLayer.next_id
        TrainModeLayer.next_id += 1

    @abstractmethod
    def get_id(self):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def get_name(self):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def forward(self, input_data: ndarray, test_model_params: dict):
        """
        Performs a forward pass of a layer, updating a test model parameters if required.
        :param input_data: is a layer input data.
        :param test_model_params: is a dict of parameters, required to build a test model.
        :return: layer forward run.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def backward(self, dout: ndarray, layer_forward_run: Cache):
        """
        Performs a backward pass of a layer, based on the layer forward run and the gradient by this layer.
        :param dout: is a gradient by this layer.
        :param layer_forward_run: is a layer forward run parameters.
        :return: gradient by it's input data and a layer backward run parameters.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def to_test(self, test_model_params: dict):
        """
        Creates a test mode layer representative based on it's weights and test model parameters.
        :param test_model_params: is a dict of a parameters, required to build a test model layers.
        :return: a test mode layer object.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def accept(self, visitor):
        raise NotImplementedError(NOT_IMPLEMENTED)


class TestModeLayer:
    """
    Test mode layer representative.
    """
    __metaclass__ = ABCMeta

    next_id = 0

    def __init__(self):
        self.id = TrainModeLayer.next_id
        TrainModeLayer.next_id += 1

    @abstractmethod
    def get_id(self):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def get_name(self):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def forward(self, input_data: ndarray):
        """
        Performs a forward pass of a layer based on the input data.
        :param input_data: is an input data of a layer.
        :return: output data of a layer.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)
