import numpy as np
from numpy import ndarray

from neural_nets.model.Layer import TrainModeLayer, TestModeLayer
from neural_nets.model.Name import Name
from neural_nets.model.Params import Params


class BatchNormTest(TestModeLayer):
    def __init__(self, input_dim: int, momentum: float):
        super().__init__()
        self.momentum = momentum

        self.weights = self.create_weights(input_dim=input_dim)
        self.params = self.create_params(input_dim=input_dim)

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.BATCH_NORM_TEST

    def get_weights(self):
        return self.weights

    def get_params(self):
        return self.params

    def forward(self, input_data: ndarray):
        output = (input_data - self.params.get(Name.RUNNING_MEAN)) / np.sqrt(
            self.params.get(name=Name.RUNNING_VARIANCE) + 1e-5)
        output = self.weights.get(name=Name.GAMMA) * output + self.weights.get(name=Name.BETA)
        return output

    @staticmethod
    def create_weights(input_dim: int):
        weights = Params()
        weights.add(name=Name.GAMMA, value=0.01 * np.random.rand(input_dim, 1))
        weights.add(name=Name.BETA, value=0.01 * np.random.rand(input_dim, 1))
        return weights

    @staticmethod
    def create_params(input_dim: int):
        params = Params()
        params.add(name=Name.RUNNING_MEAN, value=np.zeros((input_dim, 1)))
        params.add(name=Name.RUNNING_VARIANCE, value=np.zeros((input_dim, 1)))
        return params


class BatchNormTrain(TrainModeLayer):
    def __init__(self, input_dim: int, momentum: float):
        super().__init__()
        self.input_dim = input_dim
        self.momentum = momentum
        self.weights = self.create_weights(input_dim=input_dim)

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.BATCH_NORM_TRAIN

    def get_weights(self):
        return self.weights

    def get_momentum(self):
        return self.momentum

    def forward(self, input_data: ndarray, test_model_params: dict):
        layer_forward_run = Params()
        layer_forward_run.add(name=Name.INPUT, value=input_data)

        mu = np.mean(input_data, axis=1, keepdims=True)
        layer_forward_run.add(name=Name.MU, value=mu)

        var = np.var(input_data, axis=1, keepdims=True)
        layer_forward_run.add(name=Name.VAR, value=var)

        running_mean = self.momentum * test_model_params[self.id].get(name=Name.RUNNING_MEAN) + (
                1.0 - self.momentum) * mu
        test_model_params[self.id].update(name=Name.RUNNING_MEAN, value=running_mean)

        running_var = self.momentum * test_model_params[self.id].get(name=Name.RUNNING_VARIANCE) + (
                1.0 - self.momentum) * var
        test_model_params[self.id].update(name=Name.RUNNING_VARIANCE, value=running_var)

        X_norm = (input_data - mu) / np.sqrt(var + 1e-5)
        output_data = self.weights.get(name=Name.GAMMA) * X_norm + self.weights.get(name=Name.BETA)
        layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Params):
        layer_backward_run = Params()

        input_data = layer_forward_run.get(name=Name.INPUT)
        N = input_data.shape[1]

        mu = layer_forward_run.get(name=Name.MU)
        var = layer_forward_run.get(name=Name.VAR)

        dgamma = np.sum((input_data - mu) * (var + 1e-5) ** (-1. / 2.) * dout, axis=1, keepdims=True)
        layer_backward_run.add(name=Name.D_GAMMA, value=dgamma)

        dbeta = np.sum(dout, axis=1, keepdims=True)
        layer_backward_run.add(name=Name.D_BETA, value=dbeta)

        dinput = (1. / N) * self.weights.get(name=Name.GAMMA) * (var + 1e-5) ** (-1. / 2.) * (
                N * dout - np.sum(dout, axis=1, keepdims=True) - (input_data - mu) * (var + 1e-5) **
                (-1.0) * np.sum(dout * (input_data - mu), axis=1, keepdims=True))

        return dinput, layer_backward_run

    def to_test(self, test_model_params: dict):
        weights = self.weights
        layer = BatchNormTest(input_dim=self.input_dim, momentum=self.momentum)

        layer.get_weights().update(name=Name.GAMMA, value=weights.get(name=Name.GAMMA))
        layer.get_weights().update(name=Name.BETA, value=weights.get(name=Name.BETA))

        params = test_model_params.get(self.id)
        layer.get_params().update(name=Name.RUNNING_MEAN, value=params.get(Name.RUNNING_MEAN))
        layer.get_params().update(name=Name.RUNNING_VARIANCE, value=params.get(Name.RUNNING_VARIANCE))

        return layer

    def accept(self, visitor):
        visitor.visit_batch_norm(self)

    @staticmethod
    def create_weights(input_dim: int):
        weights = Params()
        weights.add(name=Name.GAMMA, value=0.01 * np.random.rand(input_dim, 1))
        weights.add(name=Name.BETA, value=0.01 * np.random.rand(input_dim, 1))
        return weights
