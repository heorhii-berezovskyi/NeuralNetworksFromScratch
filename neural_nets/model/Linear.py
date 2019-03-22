import numpy as np
from numpy import ndarray

from neural_nets.model.Layer import TrainModeLayer, TestModeLayer
from neural_nets.model.Name import Name
from neural_nets.model.Params import Params


class LinearTest(TestModeLayer):
    def __init__(self, num_of_neurons: int, input_dim: int):
        super().__init__()
        self.weights = self.create_weights(num_of_neurons=num_of_neurons, input_dim=input_dim)

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.LINEAR_TEST

    def forward(self, input_data: ndarray):
        output = np.dot(self.weights.get(name=Name.WEIGHTS), input_data) + self.weights.get(name=Name.BIASES)
        return output

    def get_weights(self):
        return self.weights

    @staticmethod
    def create_weights(num_of_neurons: int, input_dim: int):
        weights = Params()
        weights.add(name=Name.WEIGHTS, value=0.01 * np.random.rand(num_of_neurons, input_dim))
        weights.add(name=Name.BIASES, value=np.zeros((num_of_neurons, 1)))
        return weights


class LinearTrain(TrainModeLayer):
    def __init__(self, num_of_neurons: int, input_dim: int):
        super().__init__()
        self.num_of_neurons = num_of_neurons
        self.input_dim = input_dim
        self.weights = self.create_weights(num_of_neurons=num_of_neurons, input_dim=input_dim)

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.LINEAR_TRAIN

    def get_weights(self):
        return self.weights

    def forward(self, input_data: ndarray, test_model_params: dict):
        output_data = np.dot(self.weights.get(name=Name.WEIGHTS), input_data) + self.weights.get(name=Name.BIASES)
        layer_forward_run = Params()
        layer_forward_run.add(name=Name.INPUT, value=input_data)
        layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Params):
        input_data = layer_forward_run.get(name=Name.INPUT)
        layer_backward_run = Params()

        dweights = np.dot(dout, input_data.T)
        layer_backward_run.add(name=Name.D_WEIGHTS, value=dweights)

        dbiases = np.sum(dout, axis=1, keepdims=True)
        layer_backward_run.add(name=Name.D_BIASES, value=dbiases)

        dinput = np.dot(self.weights.get(name=Name.WEIGHTS).T, dout)
        return dinput, layer_backward_run

    def to_test(self, test_model_params: dict):
        weights = self.weights
        layer = LinearTest(self.num_of_neurons, self.input_dim)
        layer.get_weights().update(name=Name.WEIGHTS, value=weights.get(name=Name.WEIGHTS))
        layer.get_weights().update(name=Name.BIASES, value=weights.get(name=Name.BIASES))
        return layer

    def accept(self, visitor):
        visitor.visit_linear(self)

    @staticmethod
    def create_weights(num_of_neurons: int, input_dim: int):
        weights = Params()
        weights.add(name=Name.WEIGHTS, value=0.01 * np.random.rand(num_of_neurons, input_dim))
        weights.add(name=Name.BIASES, value=np.zeros((num_of_neurons, 1)))
        return weights
