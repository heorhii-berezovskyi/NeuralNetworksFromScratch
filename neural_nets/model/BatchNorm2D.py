import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayerWithWeights, TestModeLayerWithWeights
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TrainLayerVisitor, TestLayerVisitor
from neural_nets.optimizer.Optimizer import Optimizer


class BatchNorm2DTest(TestModeLayerWithWeights):
    name = Name.BATCH_NORM_2D_TEST

    def __init__(self, layer_id: int, weights: Cache, params: Cache):
        self.id = layer_id
        self.weights = weights
        self.params = params

    def forward(self, input_data: ndarray) -> ndarray:
        gamma, beta = self.weights.get(name=Name.GAMMA), self.weights.get(name=Name.BETA)
        running_mean, running_variance = self.params.get(Name.RUNNING_MEAN), self.params.get(name=Name.RUNNING_VAR)

        N, C, H, W = input_data.shape
        x_flat = input_data.transpose((0, 2, 3, 1)).reshape(-1, C)

        xn_flat = (x_flat - running_mean) / np.sqrt(running_variance + 1e-5)
        output_flat = gamma * xn_flat + beta

        output_data = output_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return output_data

    def content(self) -> dict:
        layer_id = BatchNorm2DTest.name.value + str(self.id)
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

        layer_id = BatchNorm2DTest.name.value + str(self.id)
        for w_name, p_name in zip([Name.GAMMA, Name.BETA], [Name.RUNNING_MEAN, Name.RUNNING_VAR]):
            w_key = layer_id + w_name.value
            w_value = all_params[w_key]
            weights.add(name=w_name, value=w_value)

            p_key = layer_id + p_name.value
            p_value = all_params[p_key]
            params.add(name=p_name, value=p_value)
        return BatchNorm2DTest(layer_id=self.id, weights=weights, params=params)

    def accept(self, visitor: TestLayerVisitor):
        visitor.visit_weighted_test(self)


class BatchNorm2DTrain(TrainModeLayerWithWeights):
    name = Name.BATCH_NORM_2D_TRAIN

    def __init__(self, layer_id: int, weights: Cache, momentum: float, optimizer: Optimizer):
        self.id = layer_id
        self.optimizer = optimizer
        self.momentum = momentum
        self.weights = weights

    def init_params(self) -> Cache:
        params = Cache()
        params.add(name=Name.RUNNING_MEAN, value=np.zeros_like(self.weights.get(name=Name.GAMMA), dtype=float))
        params.add(name=Name.RUNNING_VAR, value=np.zeros_like(self.weights.get(name=Name.GAMMA), dtype=float))
        return params

    def forward(self, input_data: ndarray, layer_forward_run: Cache) -> Cache:
        N, C, H, W = input_data.shape

        input_flat = input_data.transpose((0, 2, 3, 1)).reshape(-1, C)

        mu = np.mean(input_flat, axis=0)
        xmu = input_flat - mu
        var = np.var(input_flat, axis=0)
        sqrtvar = np.sqrt(var + 1e-5)
        ivar = 1. / sqrtvar
        xhat = xmu * ivar
        out_flat = self.weights.get(name=Name.GAMMA) * xhat + self.weights.get(name=Name.BETA)

        output_data = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)

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
        N, C, H, W = dout.shape
        dout_flat = dout.transpose((0, 2, 3, 1)).reshape(-1, C)

        N_f, D_f = dout_flat.shape
        xhat, ivar = layer_forward_run.pop(name=Name.X_HAT), layer_forward_run.pop(name=Name.IVAR)

        dbeta = np.sum(dout_flat, axis=0)
        dgamma = np.sum(xhat * dout_flat, axis=0)

        dxhat = dout_flat * self.weights.get(name=Name.GAMMA)
        dinput_flat = 1. / N_f * ivar * (N_f * dxhat - np.sum(dxhat, axis=0) - xhat * np.sum(dxhat * xhat, axis=0))
        dinput = dinput_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        layer_backward_run = Cache()
        layer_backward_run.add(name=Name.D_INPUT, value=dinput)
        layer_backward_run.add(name=Name.GAMMA, value=dgamma)
        layer_backward_run.add(name=Name.BETA, value=dbeta)
        return layer_backward_run

    def to_test(self, test_layer_params: Cache) -> TestModeLayerWithWeights:
        return BatchNorm2DTest(layer_id=self.id, weights=self.weights, params=test_layer_params)

    def optimize(self, layer_backward_run: Cache) -> TrainModeLayerWithWeights:
        new_optimizer = self.optimizer.update_memory(layer_backward_run=layer_backward_run)
        new_weights = new_optimizer.update_weights(self.weights)
        return BatchNorm2DTrain(layer_id=self.id,
                                weights=new_weights,
                                momentum=self.momentum,
                                optimizer=new_optimizer)

    def accept(self, visitor: TrainLayerVisitor):
        visitor.visit_batch_norm_2d_train(self)

    @staticmethod
    def init_weights(num_of_channels: int) -> Cache:
        weights = Cache()
        weights.add(name=Name.GAMMA, value=np.ones(num_of_channels, dtype=float))
        weights.add(name=Name.BETA, value=np.zeros(num_of_channels, dtype=float))
        return weights
