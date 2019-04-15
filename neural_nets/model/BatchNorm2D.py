import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayerWithWeights, TestModeLayerWithWeightsAndParams
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TrainLayerVisitor, TestLayerVisitor


class BatchNorm2DTest(TestModeLayerWithWeightsAndParams):
    def __init__(self, layer_id: int, weights: Cache, params: Cache):
        self.id = layer_id
        self.weights = weights
        self.params = params

    def get_id(self) -> int:
        return self.id

    def get_name(self) -> Name:
        return Name.BATCH_NORM_2D_TEST

    def get_weights(self) -> Cache:
        return self.weights

    def get_params(self) -> Cache:
        return self.params

    def forward(self, input_data: ndarray) -> ndarray:
        gamma, beta = self.weights.get(name=Name.GAMMA), self.weights.get(name=Name.BETA)
        running_mean, running_variance = self.params.get(Name.RUNNING_MEAN), self.params.get(name=Name.RUNNING_VAR)

        N, C, H, W = input_data.shape
        x_flat = input_data.transpose((0, 2, 3, 1)).reshape(-1, C)

        xn_flat = (x_flat - running_mean) / np.sqrt(running_variance + 1e-5)
        output_flat = gamma * xn_flat + beta

        output_data = output_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return output_data

    def accept(self, visitor: TestLayerVisitor):
        visitor.visit_batch_norm_test(self)


class BatchNorm2DTrain(TrainModeLayerWithWeights):
    def __init__(self, weights: Cache, momentum: float):
        super().__init__()
        self.momentum = momentum
        self.weights = weights

    def get_id(self) -> int:
        return self.id

    def get_name(self) -> Name:
        return Name.BATCH_NORM_2D_TRAIN

    def get_weights(self) -> Cache:
        return self.weights

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
        layer_backward_run.add(name=Name.D_GAMMA, value=dgamma)
        layer_backward_run.add(name=Name.D_BETA, value=dbeta)
        return layer_backward_run

    def to_test(self, test_layer_params: Cache) -> TestModeLayerWithWeightsAndParams:
        return BatchNorm2DTest(layer_id=self.id, weights=self.weights, params=test_layer_params)

    def accept(self, visitor: TrainLayerVisitor):
        visitor.visit_batch_norm_train(self)

    @classmethod
    def init_weights(cls, num_of_channels: int, momentum: float):
        weights = Cache()
        weights.add(name=Name.GAMMA, value=np.ones(num_of_channels, dtype=np.float64))
        weights.add(name=Name.BETA, value=np.zeros(num_of_channels, dtype=np.float64))
        return cls(weights, momentum=momentum)
