import numpy as np
from numpy import ndarray

from neural_nets.model.Layer import TrainModeLayer, TestModeLayer
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TestModelInitVisitor


class TestModel:
    """
    Model representative working at test mode.
    """

    def __init__(self):
        self.layers = []

    def add(self, layer: TestModeLayer):
        """
        Adds test mode layer to the model.
        :param layer: is a test mode layer representative.
        """
        self.layers.append(layer)

    def test(self, labels: ndarray, images: ndarray):
        """
        Computes the accuracy of the model over the image batch.
        :param labels: is a labels of data.
        :param images: is an image batch to compute accuracy on.
        :return: float value of accuracy in a range [0, 1].
        """
        input_data = images
        for layer in self.layers:
            output_data = layer.forward(input_data)
            input_data = output_data
        predicted_class = np.argmax(input_data, axis=1)
        accuracy = np.mean(predicted_class == labels)
        return accuracy


class TrainModel:
    """
    Model representative working at train mode.
    """

    def __init__(self):
        self.layers = []

    def get_layers(self):
        return self.layers

    def add(self, layer: TrainModeLayer):
        """
        Adds train mode layer to the model.
        :param layer: is a train mode layer.
        """
        self.layers.append(layer)

    def init_test_model_params(self):
        """
        Initializes params required for building a test model.
        :return: test model parameters dict.
        """
        visitor = TestModelInitVisitor()
        for layer in self.layers:
            layer.accept(visitor)
        return visitor.get_init_test_model_params()

    def forward(self, test_model_params: dict, images: ndarray):
        """
        Runs layer by layer in a forward mode, passing an input data and updating test model parameters.
        :param test_model_params: is a dict of parameters required for a test model.
        :param images: is a batch of training images.
        :return: model forward run – list of parameters, saved by each layer through a forward pass.
        """
        input_data = images
        model_forward_run = []
        for layer in self.layers:
            layer_forward_run = layer.forward(input_data, test_model_params)
            model_forward_run.append(layer_forward_run)
            input_data = layer_forward_run.get(Name.OUTPUT)
        return model_forward_run

    def to_test(self, test_model_params: dict):
        """
        Builds a test model based on a train model and test model parameters.
        :param test_model_params: is a dict of parameters required to build a test model and updated through a multiple
               forward pass of a train model.
        :return: a built test model.
        """
        test_model = TestModel()
        for layer in self.layers:
            test_layer = layer.to_test(test_model_params)
            test_model.add(test_layer)
        return test_model
