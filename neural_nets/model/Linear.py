import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayerWithWeights, TestModeLayerWithWeights
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TrainLayerVisitor, TestLayerVisitor
from neural_nets.optimizer.Optimizer import Optimizer


class LinearTest(TestModeLayerWithWeights):
    name = Name.LINEAR_TEST

    def __init__(self, block_name: str, weights: Cache):
        super().__init__(block_name=block_name)
        self.weights = weights

    def forward(self, input_data: ndarray) -> ndarray:
        weights, biases = self.weights.get(name=Name.WEIGHTS), self.weights.get(name=Name.BIASES)
        output_data = input_data.reshape(input_data.shape[0], -1) @ weights + biases
        return output_data

    def content(self) -> dict:
        layer_id = self.block_name + LinearTest.name.value
        result = {}
        for item_name in self.weights.get_keys():
            data_value = self.weights.get(name=item_name)
            data_key = layer_id + item_name.value
            result[data_key] = data_value
        return result

    def from_params(self, all_params):
        weights = Cache()
        layer_id = self.block_name + LinearTest.name.value
        for w_name in [Name.WEIGHTS, Name.BIASES]:
            w_key = layer_id + w_name.value
            w_value = all_params[w_key]
            weights.add(name=w_name, value=w_value)
        return LinearTest(block_name=self.block_name, weights=weights)

    def accept(self, visitor: TestLayerVisitor):
        visitor.visit_weighted_test(self)


class LinearTrain(TrainModeLayerWithWeights):
    name = Name.LINEAR_TRAIN

    def __init__(self, block_name: str, weights: Cache, optimizer: Optimizer):
        super().__init__(block_name=block_name)
        self.optimizer = optimizer
        self.weights = weights

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
        layer_backward_run.add(name=Name.WEIGHTS, value=dweights)
        layer_backward_run.add(name=Name.BIASES, value=dbiases)
        return layer_backward_run

    def to_test(self, test_layer_params: Cache) -> TestModeLayerWithWeights:
        return LinearTest(block_name=self.block_name, weights=self.weights)

    def optimize(self, layer_backward_run: Cache) -> TrainModeLayerWithWeights:
        new_optimizer = self.optimizer.update_memory(layer_backward_run=layer_backward_run)
        new_weights = new_optimizer.update_weights(self.weights)
        return LinearTrain(block_name=self.block_name,
                           weights=new_weights,
                           optimizer=new_optimizer)

    def accept(self, visitor: TrainLayerVisitor):
        visitor.visit_affine_train(self)

    @staticmethod
    def init_weights(input_dim: int, num_of_neurons: int):
        weights = Cache()
        weights.add(name=Name.WEIGHTS, value=np.random.rand(input_dim, num_of_neurons) * np.sqrt(2. / input_dim))
        weights.add(name=Name.BIASES, value=np.zeros(num_of_neurons, dtype=float))
        return weights
