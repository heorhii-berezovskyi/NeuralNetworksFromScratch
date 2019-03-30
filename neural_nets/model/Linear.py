import numpy as np
from numpy import ndarray

from neural_nets.model.Layer import TrainModeLayer, TestModeLayer
from neural_nets.model.Name import Name
from neural_nets.model.Cache import Cache


class LinearTest(TestModeLayer):
    def __init__(self, input_dim: int, num_of_neurons: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_of_neurons = num_of_neurons
        self.weights = self.create_weights(input_dim=input_dim, num_of_neurons=num_of_neurons)

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.LINEAR_TEST

    def forward(self, input_data: ndarray):
        num_of_samples = input_data.shape[0]
        reshaped_input_data = np.reshape(input_data, (num_of_samples, self.input_dim))

        weights, biases = self.weights.get(name=Name.WEIGHTS), self.weights.get(name=Name.BIASES)
        output_data = reshaped_input_data @ weights + biases
        return output_data

    def get_weights(self):
        return self.weights

    @staticmethod
    def create_weights(input_dim: int, num_of_neurons: int):
        weights = Cache()
        weights.add(name=Name.WEIGHTS, value=0.01 * np.random.rand(input_dim, num_of_neurons))
        weights.add(name=Name.BIASES, value=np.zeros(num_of_neurons))
        return weights


class LinearTrain(TrainModeLayer):
    def __init__(self, input_dim: int, num_of_neurons: int):
        super().__init__()
        self.num_of_neurons = num_of_neurons
        self.input_dim = input_dim
        self.weights = self.create_weights(input_dim=input_dim, num_of_neurons=num_of_neurons)

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.LINEAR_TRAIN

    def get_weights(self):
        return self.weights

    def forward(self, input_data: ndarray, test_model_params: dict):
        num_of_samples = input_data.shape[0]
        reshaped_input_data = np.reshape(input_data, (num_of_samples, self.input_dim))

        weights, biases = self.weights.get(name=Name.WEIGHTS), self.weights.get(name=Name.BIASES)
        output_data = reshaped_input_data @ weights + biases

        layer_forward_run = Cache()
        layer_forward_run.add(name=Name.INPUT, value=input_data)
        layer_forward_run.add(name=Name.RESHAPED_INPUT_DATA, value=reshaped_input_data)
        layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache):
        input_data = layer_forward_run.get(name=Name.INPUT)
        reshaped_input_data = layer_forward_run.get(name=Name.RESHAPED_INPUT_DATA)
        num_of_samples = input_data.shape[0]
        weights = self.weights.get(name=Name.WEIGHTS)

        dreshaped_input_data = dout @ weights.T  # N x D
        dweights = reshaped_input_data.T @ dout  # D x M
        dbiases = dout.T @ np.ones(num_of_samples)  # M x 1
        dinput_data = np.reshape(dreshaped_input_data, input_data.shape)

        layer_backward_run = Cache()
        layer_backward_run.add(name=Name.D_WEIGHTS, value=dweights)
        layer_backward_run.add(name=Name.D_BIASES, value=dbiases)
        return dinput_data, layer_backward_run

    def to_test(self, test_model_params: dict):
        weights = self.weights
        layer = LinearTest(input_dim=self.input_dim, num_of_neurons=self.num_of_neurons)
        layer.get_weights().update(name=Name.WEIGHTS, value=weights.get(name=Name.WEIGHTS))
        layer.get_weights().update(name=Name.BIASES, value=weights.get(name=Name.BIASES))
        return layer

    def accept(self, visitor):
        visitor.visit_linear(self)

    @staticmethod
    def create_weights(input_dim: int, num_of_neurons: int):
        weights = Cache()
        weights.add(name=Name.WEIGHTS, value=0.01 * np.random.rand(input_dim, num_of_neurons))
        weights.add(name=Name.BIASES, value=np.zeros(num_of_neurons))
        return weights
