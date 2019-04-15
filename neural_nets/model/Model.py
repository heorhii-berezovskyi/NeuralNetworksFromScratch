import numpy as np
from numpy import ndarray

from neural_nets.model.Layer import TrainModeLayer, TestModeLayer
from neural_nets.model.Name import Name
from neural_nets.model.TrainModelInitVisitor import TrainModelInitVisitor
from neural_nets.model.TestModelLoadVisitor import TestModelLoadVisitor
from neural_nets.model.TestModelSaveVisitor import TestModelSaveVisitor


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

    def save(self, path: str):
        visitor = TestModelSaveVisitor()
        for layer in self.layers:
            layer.accept(visitor)
        all_params = visitor.get_result()
        np.savez(path, **all_params)

    def load(self, path: str):
        all_params = np.load(path)
        visitor = TestModelLoadVisitor(all_params=all_params)
        for layer in self.layers:
            layer.accept(visitor)
        all_params.close()


class TrainModel:
    """
    Model representative working at train mode.
    """

    def __init__(self):
        self.layers = []

    def get_layers(self) -> list:
        return self.layers

    def add(self, layer: TrainModeLayer):
        """
        Adds train mode layer to the model.
        :param layer: is a train mode layer.
        """
        self.layers.append(layer)

    def init_model(self) -> list:
        """
        Initializes params required for building a test model.
        :return: list of prepared params.
        """
        visitor = TrainModelInitVisitor()
        for layer in self.layers:
            layer.accept(visitor)
        return visitor.get_result()

    def forward(self, model_forward_run: list, images: ndarray) -> tuple:
        """
        Runs layer by layer in a forward mode, passing an input data and updating test model parameters.
        :param model_forward_run: is a list of cached parameters required for an optimizer and to build a test model.
        :param images: is a batch of training images.
        :return: next model forward run and model scores.
        """
        input_data = images
        next_model_forward_run = []
        for layer, layer_forward_run in zip(self.layers, model_forward_run):
            new_layer_forward_run = layer.forward(input_data, layer_forward_run)
            input_data = new_layer_forward_run.pop(Name.OUTPUT)
            next_model_forward_run.append(new_layer_forward_run)
        scores = input_data
        return next_model_forward_run, scores

    def to_test(self, model_forward_run: list) -> TestModel:
        """
        Builds a test model based on a train model and test model parameters.
        :param model_forward_run: is a list of parameters required to build a test model and updated through a multiple
               forward pass of a train model.
        :return: a built test model.
        """
        test_model = TestModel()
        for train_layer, layer_forward_run in zip(self.layers, model_forward_run):
            test_layer = train_layer.to_test(layer_forward_run)
            test_model.add(test_layer)
        return test_model
