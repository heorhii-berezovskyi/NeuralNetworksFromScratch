import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TestModeLayer
from neural_nets.model.Name import Name


class BatchNorm1DTest(TestModeLayer):
    def __init__(self, weights: Cache, params: Cache):
        super().__init__()
        self.weights = weights
        self.params = params

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.BATCH_NORM_1D_TEST

    def get_weights(self):
        return self.weights

    def get_params(self):
        return self.params

    def forward(self, input_data: ndarray):
        running_mean, running_variance = self.params.get(Name.RUNNING_MEAN), self.params.get(name=Name.RUNNING_VARIANCE)
        gamma, beta = self.weights.get(name=Name.GAMMA), self.weights.get(name=Name.BETA)

        xn = (input_data - running_mean) / np.sqrt(running_variance + 1e-5)
        output_data = gamma * xn + beta
        return output_data


class BatchNorm1DTrain(TrainModeLayer):
    def __init__(self, input_dim: int, momentum: float):
        super().__init__()
        self.input_dim = input_dim
        self.momentum = momentum
        self.weights = self.create_weights(input_dim=input_dim)

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.BATCH_NORM_1D_TRAIN

    def get_weights(self):
        return self.weights

    def forward(self, input_data: ndarray, test_model_params: dict):
        gamma, beta = self.weights.get(name=Name.GAMMA), self.weights.get(name=Name.BETA)
        eps = 1e-5
        mu = input_data.mean(axis=0)  # Size (H,)
        var = np.var(input_data, axis=0)  # Size (H,)
        xn = (input_data - mu) * (var + eps) ** (-1. / 2.)
        output_data = gamma * xn + beta

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
        layer_forward_run.add(name=Name.INPUT, value=input_data)
        layer_forward_run.add(name=Name.MU, value=mu)
        layer_forward_run.add(name=Name.VAR, value=var)
        layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache):
        input_data = layer_forward_run.get(name=Name.INPUT)
        mu = layer_forward_run.get(name=Name.MU)
        var = layer_forward_run.get(name=Name.VAR)
        gamma = self.weights.get(name=Name.GAMMA)
        N = input_data.shape[0]

        # Centered data.
        xc = input_data - mu

        # Stable variance.
        s_var = var + 1e-5

        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(xc * s_var ** (-1. / 2.) * dout, axis=0)
        dinput = (1. / N) * gamma * s_var ** (-1. / 2.) * (
                N * dout - np.sum(dout, axis=0) - xc * s_var ** (-1.) * np.sum(dout * xc, axis=0))

        layer_backward_run = Cache()
        layer_backward_run.add(name=Name.D_GAMMA, value=dgamma)
        layer_backward_run.add(name=Name.D_BETA, value=dbeta)
        return dinput, layer_backward_run

    def to_test(self, test_model_params: dict):
        weights = self.weights
        params = test_model_params.get(self.id)
        layer = BatchNorm1DTest(weights=weights, params=params)
        return layer

    def accept(self, visitor):
        visitor.visit_batch_norm_1d(self)

    @staticmethod
    def create_weights(input_dim: int):
        weights = Cache()
        weights.add(name=Name.GAMMA, value=np.ones(input_dim))
        weights.add(name=Name.BETA, value=np.zeros(input_dim))
        return weights
