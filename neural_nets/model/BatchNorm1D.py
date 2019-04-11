import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayerWithWeights, TestModeLayerWithWeightsAndParams
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TestLayerBaseVisitor, TrainLayerBaseVisitor


class BatchNorm1DTest(TestModeLayerWithWeightsAndParams):
    def __init__(self, layer_id: int, weights: Cache, params: Cache):
        self.id = layer_id
        self.weights = weights
        self.params = params

    def get_id(self) -> int:
        return self.id

    def get_name(self) -> Name:
        return Name.BATCH_NORM_1D_TEST

    def get_weights(self) -> Cache:
        return self.weights

    def get_params(self) -> Cache:
        return self.params

    def forward(self, input_data: ndarray) -> ndarray:
        running_mean, running_variance = self.params.get(Name.RUNNING_MEAN), self.params.get(name=Name.RUNNING_VARIANCE)
        gamma, beta = self.weights.get(name=Name.GAMMA), self.weights.get(name=Name.BETA)

        xn = (input_data - running_mean) / np.sqrt(running_variance + 1e-5)
        output_data = gamma * xn + beta
        return output_data

    def accept(self, visitor: TestLayerBaseVisitor):
        visitor.visit_batch_norm_test(self)


class BatchNorm1DTrain(TrainModeLayerWithWeights):
    def __init__(self, input_dim: int, momentum: float):
        super().__init__()
        self.input_dim = input_dim
        self.momentum = momentum
        self.weights = self.create_weights(input_dim=input_dim)

    def get_id(self) -> int:
        return self.id

    def get_name(self) -> Name:
        return Name.BATCH_NORM_1D_TRAIN

    def get_weights(self):
        return self.weights

    def forward(self, input_data: ndarray, test_model_params: dict) -> Cache:
        gamma, beta = self.weights.get(name=Name.GAMMA), self.weights.get(name=Name.BETA)

        mu = np.mean(input_data, axis=0)
        xmu = input_data - mu
        var = np.var(input_data, axis=0)
        sqrtvar = np.sqrt(var + 1e-5)
        ivar = 1. / sqrtvar
        xhat = xmu * ivar
        output_data = gamma * xhat + beta

        running_mean = test_model_params[self.id].get(name=Name.RUNNING_MEAN)
        running_variance = test_model_params[self.id].get(name=Name.RUNNING_VARIANCE)

        # Update running average of mean
        running_mean *= self.momentum
        running_mean += (1. - self.momentum) * mu

        # Update running average of variance
        running_variance *= self.momentum
        running_variance += (1. - self.momentum) * var

        test_model_params[self.id].update(name=Name.RUNNING_MEAN, value=running_mean)
        test_model_params[self.id].update(name=Name.RUNNING_VARIANCE, value=running_variance)

        layer_forward_run = Cache()
        layer_forward_run.add(name=Name.X_HAT, value=xhat)
        layer_forward_run.add(name=Name.IVAR, value=ivar)
        layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache) -> tuple:
        gamma = self.weights.get(name=Name.GAMMA)
        N, D = dout.shape

        xhat, ivar = layer_forward_run.get(name=Name.X_HAT), layer_forward_run.get(name=Name.IVAR)

        dxhat = dout * gamma
        dinput = 1. / N * ivar * (N * dxhat - np.sum(dxhat, axis=0) - xhat * np.sum(dxhat * xhat, axis=0))
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(xhat * dout, axis=0)

        layer_backward_run = Cache()
        layer_backward_run.add(name=Name.D_GAMMA, value=dgamma)
        layer_backward_run.add(name=Name.D_BETA, value=dbeta)
        return dinput, layer_backward_run

    def to_test(self, test_model_params: dict) -> TestModeLayerWithWeightsAndParams:
        weights = self.weights
        params = test_model_params.get(self.id)
        layer = BatchNorm1DTest(layer_id=self.id, weights=weights, params=params)
        return layer

    def accept(self, visitor: TrainLayerBaseVisitor):
        visitor.visit_batch_norm_train(self)

    @staticmethod
    def create_weights(input_dim: int) -> Cache:
        weights = Cache()
        weights.add(name=Name.GAMMA, value=np.ones(input_dim, dtype=np.float64))
        weights.add(name=Name.BETA, value=np.zeros(input_dim, dtype=np.float64))
        return weights
