import numpy as np
from numpy import ndarray

from neural_nets.model.Layer import TrainModeLayerWithWeights, TestModeLayer
from neural_nets.model.Name import Name
from neural_nets.model.Cache import Cache


class LinearTest(TestModeLayer):
    def __init__(self, weights: Cache):
        super().__init__()
        self.weights = weights

    def get_id(self) -> int:
        return self.id

    def get_name(self) -> Name:
        return Name.LINEAR_TEST

    def forward(self, input_data: ndarray) -> ndarray:
        weights, biases = self.weights.get(name=Name.WEIGHTS), self.weights.get(name=Name.BIASES)
        output_data = input_data.reshape(input_data.shape[0], -1) @ weights + biases
        return output_data

    def get_weights(self) -> Cache:
        return self.weights


class LinearTrain(TrainModeLayerWithWeights):
    def __init__(self, input_dim: int, num_of_neurons: int):
        super().__init__()
        self.num_of_neurons = num_of_neurons
        self.input_dim = input_dim
        self.weights = self.create_weights(input_dim=input_dim, num_of_neurons=num_of_neurons)

    def get_id(self) -> int:
        return self.id

    def get_name(self) -> Name:
        return Name.LINEAR_TRAIN

    def get_weights(self) -> Cache:
        return self.weights

    def forward(self, input_data: ndarray, test_model_params: dict) -> Cache:
        weights, biases = self.weights.get(name=Name.WEIGHTS), self.weights.get(name=Name.BIASES)
        output_data = input_data.reshape(input_data.shape[0], -1) @ weights + biases

        layer_forward_run = Cache()
        layer_forward_run.add(name=Name.INPUT, value=input_data)
        layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache) -> tuple:
        input_data = layer_forward_run.get(name=Name.INPUT)
        weights = self.weights.get(name=Name.WEIGHTS)

        dinput = (dout @ weights.T).reshape(input_data.shape)
        dweights = input_data.reshape(input_data.shape[0], -1).T @ dout
        dbiases = np.sum(dout, axis=0)

        layer_backward_run = Cache()
        layer_backward_run.add(name=Name.D_WEIGHTS, value=dweights)
        layer_backward_run.add(name=Name.D_BIASES, value=dbiases)
        return dinput, layer_backward_run

    def to_test(self, test_model_params: dict) -> TestModeLayer:
        weights = self.weights
        layer = LinearTest(weights=weights)
        return layer

    def accept(self, visitor):
        visitor.visit_linear(self)

    @staticmethod
    def create_weights(input_dim: int, num_of_neurons: int) -> Cache:
        weights = Cache()
        weights.add(name=Name.WEIGHTS, value=np.random.rand(input_dim, num_of_neurons) * np.sqrt(2. / input_dim))
        weights.add(name=Name.BIASES, value=np.zeros(num_of_neurons, dtype=np.float64))
        return weights
