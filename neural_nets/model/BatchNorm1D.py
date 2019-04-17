import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayerWithWeights, TestModeLayerWithWeights
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TestLayerVisitor, TrainLayerVisitor
from neural_nets.optimizer.Optimizer import Optimizer


class BatchNorm1DTest(TestModeLayerWithWeights):
    name = Name.BATCH_NORM_1D_TEST

    def __init__(self, block_name: str, weights: Cache, params: Cache):
        super().__init__(block_name=block_name)
        self.weights = weights
        self.params = params

    def forward(self, input_data: ndarray) -> ndarray:
        running_mean, running_variance = self.params.get(Name.RUNNING_MEAN), self.params.get(name=Name.RUNNING_VAR)
        gamma, beta = self.weights.get(name=Name.GAMMA), self.weights.get(name=Name.BETA)

        xn = (input_data - running_mean) / np.sqrt(running_variance + 1e-5)
        output_data = gamma * xn + beta
        return output_data

    def content(self) -> dict:
        layer_id = self.block_name + BatchNorm1DTest.name.value
        result = {}
        for w_name, p_name in zip(self.weights.get_keys(), self.params.get_keys()):
            w = self.weights.get(name=w_name)
            w_key = layer_id + w_name.value
            result[w_key] = w

            p = self.params.get(name=p_name)
            p_key = layer_id + p_name.value
            result[p_key] = p
        return result

    def from_params(self, all_params):
        weights = Cache()
        params = Cache()

        layer_id = self.block_name + BatchNorm1DTest.name.value
        for w_name, p_name in zip([Name.GAMMA, Name.BETA], [Name.RUNNING_MEAN, Name.RUNNING_VAR]):
            w_key = layer_id + w_name.value
            w_value = all_params[w_key]
            weights.add(name=w_name, value=w_value)

            p_key = layer_id + p_name.value
            p_value = all_params[p_key]
            params.add(name=p_name, value=p_value)
        return BatchNorm1DTest(block_name=self.block_name, weights=weights, params=params)

    def accept(self, visitor: TestLayerVisitor):
        visitor.visit_weighted_test(self)


class BatchNorm1DTrain(TrainModeLayerWithWeights):
    name = Name.BATCH_NORM_1D_TRAIN

    def __init__(self, block_name: str, weights: Cache, momentum: float, optimizer: Optimizer):
        super().__init__(block_name=block_name)
        self.weights = weights
        self.momentum = momentum
        self.optimizer = optimizer

    def init_params(self) -> Cache:
        params = Cache()
        params.add(name=Name.RUNNING_MEAN, value=np.zeros_like(self.weights.get(name=Name.GAMMA), dtype=float))
        params.add(name=Name.RUNNING_VAR, value=np.zeros_like(self.weights.get(name=Name.GAMMA), dtype=float))
        return params

    def forward(self, input_data: ndarray, layer_forward_run: Cache) -> Cache:
        mu = np.mean(input_data, axis=0)
        xmu = input_data - mu
        var = np.var(input_data, axis=0)
        sqrtvar = np.sqrt(var + 1e-5)
        ivar = 1. / sqrtvar
        xhat = xmu * ivar
        output_data = self.weights.get(name=Name.GAMMA) * xhat + self.weights.get(name=Name.BETA)

        running_mean = layer_forward_run.pop(name=Name.RUNNING_MEAN) * self.momentum + (1.0 - self.momentum) * mu
        running_variance = layer_forward_run.pop(name=Name.RUNNING_VAR) * self.momentum + (1.0 - self.momentum) * var

        new_layer_forward_run = Cache()
        new_layer_forward_run.add(name=Name.X_HAT, value=xhat)
        new_layer_forward_run.add(name=Name.IVAR, value=ivar)
        new_layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        new_layer_forward_run.add(name=Name.RUNNING_MEAN, value=running_mean)
        new_layer_forward_run.add(name=Name.RUNNING_VAR, value=running_variance)
        return new_layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache) -> Cache:
        N, D = dout.shape
        xhat, ivar = layer_forward_run.pop(name=Name.X_HAT), layer_forward_run.pop(name=Name.IVAR)

        dxhat = dout * self.weights.get(name=Name.GAMMA)
        dinput = 1. / N * ivar * (N * dxhat - np.sum(dxhat, axis=0) - xhat * np.sum(dxhat * xhat, axis=0))
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(xhat * dout, axis=0)

        layer_backward_run = Cache()
        layer_backward_run.add(name=Name.D_INPUT, value=dinput)
        layer_backward_run.add(name=Name.GAMMA, value=dgamma)
        layer_backward_run.add(name=Name.BETA, value=dbeta)
        return layer_backward_run

    def to_test(self, test_layer_params: Cache) -> TestModeLayerWithWeights:
        return BatchNorm1DTest(block_name=self.block_name, weights=self.weights, params=test_layer_params)

    def optimize(self, layer_backward_run: Cache) -> TrainModeLayerWithWeights:
        new_optimizer = self.optimizer.update_memory(layer_backward_run=layer_backward_run)
        new_weights = new_optimizer.update_weights(self.weights)
        return BatchNorm1DTrain(weights=new_weights,
                                momentum=self.momentum,
                                optimizer=new_optimizer,
                                block_name=self.block_name)

    def accept(self, visitor: TrainLayerVisitor):
        visitor.visit_batch_norm_1d_train(self)

    @staticmethod
    def init_weights(input_dim: int) -> Cache:
        weights = Cache()
        weights.add(name=Name.GAMMA, value=np.ones(input_dim, dtype=float))
        weights.add(name=Name.BETA, value=np.zeros(input_dim, dtype=float))
        return weights
