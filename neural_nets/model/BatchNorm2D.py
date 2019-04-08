import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayerWithWeights, TestModeLayer
from neural_nets.model.Name import Name


class BatchNorm2DTest(TestModeLayer):
    def __init__(self, weights: Cache, params: Cache):
        super().__init__()
        self.weights = weights
        self.params = params

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.BATCH_NORM_2D_TEST

    def get_weights(self):
        return self.weights

    def get_params(self):
        return self.params

    def forward(self, input_data: ndarray):
        gamma, beta = self.weights.get(name=Name.GAMMA), self.weights.get(name=Name.BETA)
        running_mean, running_variance = self.params.get(Name.RUNNING_MEAN), self.params.get(name=Name.RUNNING_VARIANCE)

        N, C, H, W = input_data.shape
        x_flat = input_data.transpose((0, 2, 3, 1)).reshape(-1, C)

        xn_flat = (x_flat - running_mean) / np.sqrt(running_variance + 1e-5)
        output_flat = gamma * xn_flat + beta

        output_data = output_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return output_data


class BatchNorm2DTrain(TrainModeLayerWithWeights):
    def __init__(self, num_of_channels: int, momentum: float):
        super().__init__()
        self.num_of_channels = num_of_channels
        self.momentum = momentum
        self.weights = self.create_weights(num_of_channels=num_of_channels)

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.BATCH_NORM_2D_TRAIN

    def get_num_of_channels(self):
        return self.num_of_channels

    def get_weights(self):
        return self.weights

    def forward(self, input_data: ndarray, test_model_params: dict):
        gamma, beta = self.weights.get(name=Name.GAMMA), self.weights.get(name=Name.BETA)
        N, C, H, W = input_data.shape

        input_flat = input_data.transpose((0, 2, 3, 1)).reshape(-1, C)

        mu = np.mean(input_flat, axis=0)
        xmu = input_flat - mu
        var = np.var(input_flat, axis=0)
        sqrtvar = np.sqrt(var + 1e-5)
        ivar = 1. / sqrtvar
        xhat = xmu * ivar
        out_flat = gamma * xhat + beta

        output_data = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        running_mean = test_model_params[self.id].get(name=Name.RUNNING_MEAN)
        running_variance = test_model_params[self.id].get(name=Name.RUNNING_VARIANCE)

        # Update running average of mean
        running_mean *= self.momentum
        running_mean += (1.0 - self.momentum) * mu

        # Update running average of variance
        running_variance *= self.momentum
        running_variance += (1.0 - self.momentum) * var

        test_model_params[self.id].update(name=Name.RUNNING_MEAN, value=running_mean)
        test_model_params[self.id].update(name=Name.RUNNING_VARIANCE, value=running_variance)

        layer_forward_run = Cache()
        layer_forward_run.add(name=Name.X_HAT, value=xhat)
        layer_forward_run.add(name=Name.IVAR, value=ivar)
        layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache):
        N, C, H, W = dout.shape
        dout_flat = dout.transpose((0, 2, 3, 1)).reshape(-1, C)

        gamma = self.weights.get(name=Name.GAMMA)
        N_f, D_f = dout_flat.shape
        xhat, ivar = layer_forward_run.get(name=Name.X_HAT), layer_forward_run.get(name=Name.IVAR)

        dxhat = dout_flat * gamma
        dinput_flat = 1. / N_f * ivar * (N_f * dxhat - np.sum(dxhat, axis=0) - xhat * np.sum(dxhat * xhat, axis=0))
        dbeta = np.sum(dout_flat, axis=0)
        dgamma = np.sum(xhat * dout_flat, axis=0)

        dinput = dinput_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        layer_backward_run = Cache()
        layer_backward_run.add(name=Name.D_GAMMA, value=dgamma)
        layer_backward_run.add(name=Name.D_BETA, value=dbeta)
        return dinput, layer_backward_run

    def to_test(self, test_model_params: dict):
        weights = self.weights
        params = test_model_params.get(self.id)
        layer = BatchNorm2DTest(weights=weights, params=params)
        return layer

    def accept(self, visitor):
        visitor.visit_batch_norm_2d(self)

    @staticmethod
    def create_weights(num_of_channels: int):
        weights = Cache()
        weights.add(name=Name.GAMMA, value=np.ones(num_of_channels))
        weights.add(name=Name.BETA, value=np.zeros(num_of_channels))
        return weights
