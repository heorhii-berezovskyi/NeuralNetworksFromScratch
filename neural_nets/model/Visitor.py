from abc import ABCMeta, abstractmethod

import numpy as np

from neural_nets.model.BatchNorm import BatchNormTrain
from neural_nets.model.Cache import Cache
from neural_nets.model.Conv2D import Conv2DTrain
from neural_nets.model.Linear import LinearTrain
from neural_nets.model.Name import Name
from neural_nets.model.Relu import ReluTrain

NOT_IMPLEMENTED = "You should implement this."


class Visitor:
    """
    Visitor design pattern representative. Visits each type of train mode model layers.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def visit_linear(self, layer: LinearTrain):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_conv2d(self, layer: Conv2DTrain):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_relu(self, layer: ReluTrain):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_batch_norm(self, layer: BatchNormTrain):
        raise NotImplementedError(NOT_IMPLEMENTED)


class TestModelInitVisitor(Visitor):
    """
    Initializes a dict of parameters required to build a test model.
    """

    def __init__(self):
        self.result = {}

    def visit_linear(self, layer: LinearTrain):
        pass

    def visit_conv2d(self, layer: Conv2DTrain):
        pass

    def visit_relu(self, layer: ReluTrain):
        pass

    def visit_batch_norm(self, layer: BatchNormTrain):
        params = Cache()
        params.add(name=Name.RUNNING_MEAN, value=np.zeros_like(layer.get_weights().get(Name.GAMMA)))
        params.add(name=Name.RUNNING_VARIANCE, value=np.zeros_like(layer.get_weights().get(Name.GAMMA)))
        self.result[layer.get_id()] = params

    def get_init_test_model_params(self):
        return self.result


class SGDWeightsUpdateVisitor(Visitor):
    """
    Updates weights on each type of train mode model layers through a stochastic gradient update procedure.
    """

    def __init__(self, learning_rate: float, model_backward_run: list):
        self.lr = learning_rate
        self.model_backward_run = model_backward_run

    def visit_linear(self, layer: LinearTrain):
        layer_backward_run = self.model_backward_run.pop()
        layer_weights = layer.get_weights()

        weights = layer_weights.get(name=Name.WEIGHTS)
        biases = layer_weights.get(name=Name.BIASES)

        dweights = layer_backward_run.get(Name.D_WEIGHTS)
        dbiases = layer_backward_run.get(Name.D_BIASES)

        weights -= self.lr * dweights
        biases -= self.lr * dbiases

        layer.get_weights().update(name=Name.WEIGHTS, value=weights)
        layer.get_weights().update(name=Name.BIASES, value=biases)

    def visit_conv2d(self, layer: Conv2DTrain):
        layer_backward_run = self.model_backward_run.pop()
        layer_weights = layer.get_weights()

        weights = layer_weights.get(name=Name.WEIGHTS)
        biases = layer_weights.get(name=Name.BIASES)

        dweights = layer_backward_run.get(Name.D_WEIGHTS)
        dbiases = layer_backward_run.get(Name.D_BIASES)

        weights -= self.lr * dweights
        biases -= self.lr * dbiases

        layer.get_weights().update(name=Name.WEIGHTS, value=weights)
        layer.get_weights().update(name=Name.BIASES, value=biases)

    def visit_relu(self, layer: ReluTrain):
        self.model_backward_run.pop()

    def visit_batch_norm(self, layer: BatchNormTrain):
        layer_backward_run = self.model_backward_run.pop()
        layer_weights = layer.get_weights()

        gamma = layer_weights.get(name=Name.GAMMA)
        beta = layer_weights.get(name=Name.BETA)

        dgamma = layer_backward_run.get(Name.D_GAMMA)
        dbeta = layer_backward_run.get(Name.D_BETA)

        gamma -= self.lr * dgamma
        beta -= self.lr * dbeta

        layer.get_weights().update(name=Name.GAMMA, value=gamma)
        layer.get_weights().update(name=Name.BETA, value=beta)


class SGDMomentumParamsInitVisitor(Visitor):
    def __init__(self):
        self.params = {}

    def visit_linear(self, layer: LinearTrain):
        layer_velocity = Cache()
        layer_velocity.add(name=Name.V_WEIGHTS, value=np.zeros_like(layer.get_weights().get(name=Name.WEIGHTS)))
        layer_velocity.add(name=Name.V_BIASES, value=np.zeros_like(layer.get_weights().get(name=Name.BIASES)))
        self.params[layer.get_id()] = layer_velocity

    def visit_conv2d(self, layer: Conv2DTrain):
        layer_velocity = Cache()
        layer_velocity.add(name=Name.V_WEIGHTS, value=np.zeros_like(layer.get_weights().get(name=Name.WEIGHTS)))
        layer_velocity.add(name=Name.V_BIASES, value=np.zeros_like(layer.get_weights().get(name=Name.BIASES)))
        self.params[layer.get_id()] = layer_velocity

    def visit_relu(self, layer: ReluTrain):
        pass

    def visit_batch_norm(self, layer: BatchNormTrain):
        layer_velocity = Cache()
        layer_velocity.add(name=Name.V_GAMMA, value=np.zeros_like(layer.get_weights().get(name=Name.GAMMA)))
        layer_velocity.add(name=Name.V_BETA, value=np.zeros_like(layer.get_weights().get(name=Name.BETA)))
        self.params[layer.get_id()] = layer_velocity

    def get_velocity_params(self):
        return self.params


class SGDMomentumWeightsUpdateVisitor(Visitor):
    def __init__(self, learning_rate: float, mu: float, model_backward_run: list, velocity_params: dict):
        self.mu = mu
        self.lr = learning_rate
        self.model_backward_run = model_backward_run
        self.velocity_params = velocity_params

    def visit_linear(self, layer: LinearTrain):
        layer_id = layer.get_id()

        layer_backward_run = self.model_backward_run.pop()
        layer_weights = layer.get_weights()

        weights = layer_weights.get(name=Name.WEIGHTS)
        biases = layer_weights.get(name=Name.BIASES)

        dweights = layer_backward_run.get(Name.D_WEIGHTS)
        dbiases = layer_backward_run.get(Name.D_BIASES)

        v_weights = self.velocity_params[layer_id].get(Name.V_WEIGHTS)
        v_weights = self.mu * v_weights - self.lr * dweights
        self.velocity_params[layer_id].update(Name.V_WEIGHTS, v_weights)
        weights += v_weights

        v_biases = self.velocity_params[layer_id].get(Name.V_BIASES)
        v_biases = self.mu * v_biases - self.lr * dbiases
        self.velocity_params[layer_id].update(Name.V_BIASES, v_biases)
        biases += v_biases

        layer.get_weights().update(name=Name.WEIGHTS, value=weights)
        layer.get_weights().update(name=Name.BIASES, value=biases)

    def visit_conv2d(self, layer: Conv2DTrain):
        layer_id = layer.get_id()

        layer_backward_run = self.model_backward_run.pop()
        layer_weights = layer.get_weights()

        weights = layer_weights.get(name=Name.WEIGHTS)
        biases = layer_weights.get(name=Name.BIASES)

        dweights = layer_backward_run.get(Name.D_WEIGHTS)
        dbiases = layer_backward_run.get(Name.D_BIASES)

        v_weights = self.velocity_params[layer_id].get(Name.V_WEIGHTS)
        v_weights = self.mu * v_weights - self.lr * dweights
        self.velocity_params[layer_id].update(Name.V_WEIGHTS, v_weights)
        weights += v_weights

        v_biases = self.velocity_params[layer_id].get(Name.V_BIASES)
        v_biases = self.mu * v_biases - self.lr * dbiases
        self.velocity_params[layer_id].update(Name.V_BIASES, v_biases)
        biases += v_biases

        layer.get_weights().update(name=Name.WEIGHTS, value=weights)
        layer.get_weights().update(name=Name.BIASES, value=biases)

    def visit_relu(self, layer: ReluTrain):
        self.model_backward_run.pop()

    def visit_batch_norm(self, layer: BatchNormTrain):
        layer_id = layer.get_id()

        layer_backward_run = self.model_backward_run.pop()
        layer_weights = layer.get_weights()

        gamma = layer_weights.get(name=Name.GAMMA)
        beta = layer_weights.get(name=Name.BETA)

        dgamma = layer_backward_run.get(Name.D_GAMMA)
        dbeta = layer_backward_run.get(Name.D_BETA)

        v_gamma = self.velocity_params[layer_id].get(Name.V_GAMMA)
        v_gamma = self.mu * v_gamma - self.lr * dgamma
        self.velocity_params[layer_id].update(Name.V_GAMMA, v_gamma)
        gamma += v_gamma

        v_beta = self.velocity_params[layer_id].get(Name.V_BETA)
        v_beta = self.mu * v_beta - self.lr * dbeta
        self.velocity_params[layer_id].update(Name.V_BETA, v_beta)
        beta += v_beta

        layer.get_weights().update(name=Name.GAMMA, value=gamma)
        layer.get_weights().update(name=Name.BETA, value=beta)


class SGDNesterovMomentumParamsInitVisitor(Visitor):
    def __init__(self):
        self.params = {}

    def visit_linear(self, layer: LinearTrain):
        layer_velocity = Cache()
        layer_velocity.add(name=Name.V_WEIGHTS, value=np.zeros_like(layer.get_weights().get(name=Name.WEIGHTS)))
        layer_velocity.add(name=Name.V_WEIGHTS_PREV, value=np.zeros_like(layer.get_weights().get(name=Name.WEIGHTS)))

        layer_velocity.add(name=Name.V_BIASES, value=np.zeros_like(layer.get_weights().get(name=Name.BIASES)))
        layer_velocity.add(name=Name.V_BIASES_PREV, value=np.zeros_like(layer.get_weights().get(name=Name.BIASES)))
        self.params[layer.get_id()] = layer_velocity

    def visit_conv2d(self, layer: Conv2DTrain):
        layer_velocity = Cache()
        layer_velocity.add(name=Name.V_WEIGHTS, value=np.zeros_like(layer.get_weights().get(name=Name.WEIGHTS)))
        layer_velocity.add(name=Name.V_WEIGHTS_PREV, value=np.zeros_like(layer.get_weights().get(name=Name.WEIGHTS)))

        layer_velocity.add(name=Name.V_BIASES, value=np.zeros_like(layer.get_weights().get(name=Name.BIASES)))
        layer_velocity.add(name=Name.V_BIASES_PREV, value=np.zeros_like(layer.get_weights().get(name=Name.BIASES)))
        self.params[layer.get_id()] = layer_velocity

    def visit_relu(self, layer: ReluTrain):
        pass

    def visit_batch_norm(self, layer: BatchNormTrain):
        layer_velocity = Cache()
        layer_velocity.add(name=Name.V_GAMMA, value=np.zeros_like(layer.get_weights().get(name=Name.GAMMA)))
        layer_velocity.add(name=Name.V_GAMMA_PREV, value=np.zeros_like(layer.get_weights().get(name=Name.GAMMA)))

        layer_velocity.add(name=Name.V_BETA, value=np.zeros_like(layer.get_weights().get(name=Name.BETA)))
        layer_velocity.add(name=Name.V_BETA_PREV, value=np.zeros_like(layer.get_weights().get(name=Name.BETA)))
        self.params[layer.get_id()] = layer_velocity

    def get_velocity_params(self):
        return self.params


class SGDNesterovMomentumWeightsUpdateVisitor(Visitor):
    def __init__(self, learning_rate: float, mu: float, model_backward_run: list, velocity_params: dict):
        self.lr = learning_rate
        self.mu = mu
        self.model_backward_run = model_backward_run
        self.velocity_params = velocity_params

    def visit_linear(self, layer: LinearTrain):
        layer_id = layer.get_id()

        layer_backward_run = self.model_backward_run.pop()
        layer_weights = layer.get_weights()

        weights = layer_weights.get(name=Name.WEIGHTS)
        biases = layer_weights.get(name=Name.BIASES)

        dweights = layer_backward_run.get(Name.D_WEIGHTS)
        dbiases = layer_backward_run.get(Name.D_BIASES)

        v_weights = self.velocity_params[layer_id].get(Name.V_WEIGHTS)
        self.velocity_params[layer_id].update(Name.V_WEIGHTS_PREV, v_weights)
        v_weights = self.mu * v_weights - self.lr * dweights
        self.velocity_params[layer_id].update(Name.V_WEIGHTS, v_weights)
        weights += -self.mu * self.velocity_params[layer_id].get(Name.V_WEIGHTS_PREV) + (1.0 + self.mu) * v_weights

        v_biases = self.velocity_params[layer_id].get(Name.V_BIASES)
        self.velocity_params[layer_id].update(Name.V_BIASES_PREV, v_biases)
        v_biases = self.mu * v_biases - self.lr * dbiases
        self.velocity_params[layer_id].update(Name.V_BIASES, v_biases)
        biases += -self.mu * self.velocity_params[layer_id].get(Name.V_BIASES_PREV) + (1.0 + self.mu) * v_biases

        layer.get_weights().update(name=Name.WEIGHTS, value=weights)
        layer.get_weights().update(name=Name.BIASES, value=biases)

    def visit_conv2d(self, layer: Conv2DTrain):
        layer_id = layer.get_id()

        layer_backward_run = self.model_backward_run.pop()
        layer_weights = layer.get_weights()

        weights = layer_weights.get(name=Name.WEIGHTS)
        biases = layer_weights.get(name=Name.BIASES)

        dweights = layer_backward_run.get(Name.D_WEIGHTS)
        dbiases = layer_backward_run.get(Name.D_BIASES)

        v_weights = self.velocity_params[layer_id].get(Name.V_WEIGHTS)
        self.velocity_params[layer_id].update(Name.V_WEIGHTS_PREV, v_weights)
        v_weights = self.mu * v_weights - self.lr * dweights
        self.velocity_params[layer_id].update(Name.V_WEIGHTS, v_weights)
        weights += -self.mu * self.velocity_params[layer_id].get(Name.V_WEIGHTS_PREV) + (1.0 + self.mu) * v_weights

        v_biases = self.velocity_params[layer_id].get(Name.V_BIASES)
        self.velocity_params[layer_id].update(Name.V_BIASES_PREV, v_biases)
        v_biases = self.mu * v_biases - self.lr * dbiases
        self.velocity_params[layer_id].update(Name.V_BIASES, v_biases)
        biases += -self.mu * self.velocity_params[layer_id].get(Name.V_BIASES_PREV) + (1.0 + self.mu) * v_biases

        layer.get_weights().update(name=Name.WEIGHTS, value=weights)
        layer.get_weights().update(name=Name.BIASES, value=biases)

    def visit_relu(self, layer: ReluTrain):
        self.model_backward_run.pop()

    def visit_batch_norm(self, layer: BatchNormTrain):
        layer_id = layer.get_id()

        layer_backward_run = self.model_backward_run.pop()
        layer_weights = layer.get_weights()

        gamma = layer_weights.get(name=Name.GAMMA)
        beta = layer_weights.get(name=Name.BETA)

        dgamma = layer_backward_run.get(Name.D_GAMMA)
        dbeta = layer_backward_run.get(Name.D_BETA)

        v_gamma = self.velocity_params[layer_id].get(Name.V_GAMMA)
        self.velocity_params[layer_id].update(Name.V_GAMMA_PREV, v_gamma)
        v_gamma = self.mu * v_gamma - self.lr * dgamma
        self.velocity_params[layer_id].update(Name.V_GAMMA, v_gamma)
        gamma += -self.mu * self.velocity_params[layer_id].get(Name.V_GAMMA_PREV) + (1.0 + self.mu) * v_gamma

        v_beta = self.velocity_params[layer_id].get(Name.V_BETA)
        self.velocity_params[layer_id].update(Name.V_BETA_PREV, v_beta)
        v_beta = self.mu * v_beta - self.lr * dbeta
        self.velocity_params[layer_id].update(Name.V_BETA, v_beta)
        beta += -self.mu * self.velocity_params[layer_id].get(Name.V_BETA_PREV) + (1.0 + self.mu) * v_beta

        layer.get_weights().update(name=Name.GAMMA, value=gamma)
        layer.get_weights().update(name=Name.BETA, value=beta)
