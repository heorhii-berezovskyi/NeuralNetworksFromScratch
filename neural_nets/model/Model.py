import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Loss import Loss
from neural_nets.model.Name import Name
from neural_nets.model.TrainModelInitVisitor import TrainModelInitVisitor
from neural_nets.model.TrainModelLoadVisitor import TrainModelLoadVisitor
from neural_nets.model.TrainModelSaveVisitor import TrainModelSaveVisitor
from neural_nets.optimizer.Optimizer import WeightsUpdateVisitor


class TestModel:
    """
    Model representative working at test mode.
    """

    def __init__(self, layers: list):
        self.layers = layers

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

    def predict(self, image: ndarray):
        if len(image.shape) == 3:
            C, H, W = image.shape
            image = image.reshape((1, C, H, W))
        input_data = image
        for layer in self.layers:
            output_data = layer.forward(input_data)
            input_data = output_data
        predicted_class = np.argmax(input_data)
        return predicted_class


class TrainModel:
    """
    Model representative working at train mode.
    """

    def __init__(self, layers: list):
        self.layers = layers

    def init_model(self) -> list:
        """
        Initializes params required for building a test model.
        :return: list of prepared params.
        """
        visitor = TrainModelInitVisitor()
        for layer in self.layers:
            layer.accept(visitor)
        initial_model_forward_run = visitor.get_result()
        return initial_model_forward_run

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

    def backward(self, loss_function: Loss, model_forward_run: list, loss_run: Cache) -> list:
        """
        Performs backward pass of a train model based on the loss function, model forward run and loss function run.
        :param loss_function: is a loss function.
        :param model_forward_run: is a list of model forward run parameters.
        :param loss_run: is an object storing loss function run parameters.
        :return: model backward run.
        """
        dout = loss_function.eval_gradient(loss_run=loss_run)
        model_backward_run = []
        for layer, layer_forward_run in zip(reversed(self.layers), reversed(model_forward_run)):
            layer_backward_run = layer.backward(dout, layer_forward_run)
            dout = layer_backward_run.pop(Name.D_INPUT)
            model_backward_run.append(layer_backward_run)
        return model_backward_run

    def optimize(self, model_backward_run: list):
        model_backward_run.reverse()
        visitor = WeightsUpdateVisitor(model_backward_run=model_backward_run)
        for layer in reversed(self.layers):
            layer.accept(visitor)
        return TrainModel(visitor.get_result())

    def to_test(self, model_forward_run: list) -> TestModel:
        """
        Builds a test model based on a train model and test model parameters.
        :param model_forward_run: is a list of parameters required to build a test model and updated through a multiple
               forward pass of a train model.
        :return: a built test model.
        """
        test_layers = []
        for train_layer, layer_forward_run in zip(self.layers, model_forward_run.copy()):
            test_layer = train_layer.to_test(layer_forward_run)
            test_layers.append(test_layer)
        return TestModel(layers=test_layers)

    def save(self, path: str, model_forward_run: list):
        visitor = TrainModelSaveVisitor(model_forward_run=model_forward_run.copy())
        for layer in reversed(self.layers):
            layer.accept(visitor)
        all_params = visitor.get_result()
        np.savez(path, **all_params)

    def load(self, path: str) -> tuple:
        all_params = np.load(path)
        visitor = TrainModelLoadVisitor(all_params=all_params)
        for layer in self.layers:
            layer.accept(visitor)
        all_params.close()
        layers, model_forward_run = visitor.get_result()
        return TrainModel(layers=layers), model_forward_run
