import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayerWithWeights, TestModeLayerWithWeights
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TrainLayerVisitor, TestLayerVisitor


class LinearTest(TestModeLayerWithWeights):
    def __init__(self, layer_id: int, weights: Cache):
        self.id = layer_id
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

    def accept(self, visitor: TestLayerVisitor):
        visitor.visit_affine_test(self)


class LinearTrain(TrainModeLayerWithWeights):
    def __init__(self, weights: Cache):
        super().__init__()
        self.weights = weights

    def get_id(self) -> int:
        return self.id

    def get_name(self) -> Name:
        return Name.LINEAR_TRAIN

    def get_weights(self) -> Cache:
        return self.weights

    def forward(self, input_data: ndarray, layer_forward_run: Cache) -> Cache:
        weights, biases = self.weights.get(name=Name.WEIGHTS), self.weights.get(name=Name.BIASES)
        output_data = input_data.reshape(input_data.shape[0], -1) @ weights + biases

        new_layer_forward_run = Cache()
        new_layer_forward_run.add(name=Name.INPUT, value=input_data)
        new_layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return new_layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache) -> Cache:
        input_data = layer_forward_run.pop(name=Name.INPUT)
        weights = self.weights.get(name=Name.WEIGHTS)

        dinput = (dout @ weights.T).reshape(input_data.shape)
        dweights = input_data.reshape(input_data.shape[0], -1).T @ dout
        dbiases = np.sum(dout, axis=0)

        layer_backward_run = Cache()
        layer_backward_run.add(name=Name.D_INPUT, value=dinput)
        layer_backward_run.add(name=Name.D_WEIGHTS, value=dweights)
        layer_backward_run.add(name=Name.D_BIASES, value=dbiases)
        return layer_backward_run

    def to_test(self, test_layer_params: Cache) -> TestModeLayerWithWeights:
        return LinearTest(layer_id=self.id, weights=self.weights)

    def accept(self, visitor: TrainLayerVisitor):
        visitor.visit_affine_train(self)

    @classmethod
    def init_weights(cls, input_dim: int, num_of_neurons: int):
        weights = Cache()
        weights.add(name=Name.WEIGHTS, value=np.random.rand(input_dim, num_of_neurons) * np.sqrt(2. / input_dim))
        weights.add(name=Name.BIASES, value=np.zeros(num_of_neurons, dtype=np.float64))
        return cls(weights)
