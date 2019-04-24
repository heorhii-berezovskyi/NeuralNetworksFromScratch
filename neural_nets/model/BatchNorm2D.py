import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayerWithWeights, TestModeLayer
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TrainLayerVisitor
from neural_nets.optimizer.Adam import Adam
from neural_nets.optimizer.Optimizer import Optimizer


class BatchNorm2DTest(TestModeLayer):

    def __init__(self, weights: Cache, params: Cache):
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


class BatchNorm2DTrain(TrainModeLayerWithWeights):
    name = Name.BATCH_NORM_2D_TRAIN

    def __init__(self, block_name: str, weights: Cache, momentum: float, optimizer: Optimizer):
        super().__init__(block_name=block_name)
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

    def to_test(self, layer_forward_run: Cache) -> TestModeLayer:
        params = Cache()
        params.add(name=Name.RUNNING_MEAN, value=layer_forward_run.get(name=Name.RUNNING_MEAN))
        params.add(name=Name.RUNNING_VAR, value=layer_forward_run.get(name=Name.RUNNING_VAR))
        return BatchNorm2DTest(weights=self.weights,
                               params=params)

    def optimize(self, layer_backward_run: Cache) -> TrainModeLayerWithWeights:
        new_optimizer = self.optimizer.update_memory(layer_backward_run=layer_backward_run)
        new_weights = new_optimizer.update_weights(self.weights)
        return BatchNorm2DTrain(block_name=self.block_name,
                                weights=new_weights,
                                momentum=self.momentum,
                                optimizer=new_optimizer)

    def content(self, layer_forward_run: Cache) -> dict:
        layer_id = self.block_name + BatchNorm2DTrain.name.value
        result = {}

        for w_name in self.weights.get_keys():
            w_value = self.weights.get(name=w_name)
            w_key = layer_id + w_name.value
            result[w_key] = w_value

        for p_name in [Name.RUNNING_MEAN, Name.RUNNING_VAR]:
            p_value = layer_forward_run.get(name=p_name)
            p_key = layer_id + p_name.value
            result[p_key] = p_value

        optimizer_content = self.optimizer.memory_content()
        result = {**result.copy(), **optimizer_content}
        return result

    def from_params(self, all_params) -> tuple:
        layer_id = self.block_name + BatchNorm2DTrain.name.value

        weights = Cache()
        for w_name in self.weights.get_keys():
            w_key = layer_id + w_name.value
            w_value = all_params[w_key]
            weights.add(name=w_name, value=w_value)

        params = Cache()
        for p_name in [Name.RUNNING_MEAN, Name.RUNNING_VAR]:
            p_key = layer_id + p_name.value
            p_value = all_params[p_key]
            params.add(name=p_name, value=p_value)

        new_optimizer = self.optimizer.from_params(all_params=all_params)

        return BatchNorm2DTrain(block_name=self.block_name,
                                momentum=self.momentum,
                                weights=weights,
                                optimizer=new_optimizer), params

    def with_optimizer(self, optimizer_class):
        return BatchNorm2DTrain(block_name=self.block_name,
                                weights=self.weights,
                                momentum=self.momentum,
                                optimizer=optimizer_class.init_memory(
                                    layer_id=self.block_name + BatchNorm2DTrain.name.value,
                                    weights=self.weights))

    def accept(self, visitor: TrainLayerVisitor):
        visitor.visit_batch_norm_2d_train(self)

    @staticmethod
    def _init_weights(num_of_channels: int) -> Cache:
        weights = Cache()
        weights.add(name=Name.GAMMA, value=np.ones(num_of_channels, dtype=float))
        weights.add(name=Name.BETA, value=np.zeros(num_of_channels, dtype=float))
        return weights

    @classmethod
    def init(cls, block_name: str, num_of_channels: int, momentum: float, optimizer_class=Adam):
        weights = BatchNorm2DTrain._init_weights(num_of_channels=num_of_channels)
        optimizer_instance = optimizer_class.init_memory(layer_id=block_name + BatchNorm2DTrain.name.value,
                                                         weights=weights)
        return cls(block_name=block_name,
                   weights=weights,
                   momentum=momentum,
                   optimizer=optimizer_instance)
