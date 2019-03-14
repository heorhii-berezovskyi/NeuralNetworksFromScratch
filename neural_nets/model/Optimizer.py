from abc import ABCMeta, abstractmethod

import numpy as np

NOT_IMPLEMENTED = "You should implement this."


class Optimizer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(self):
        raise NotImplementedError(NOT_IMPLEMENTED)


class SGD(Optimizer):
    def __init__(self, layers: list, learning_rate: float):
        self.layers = layers
        self.lr = learning_rate

    def step(self):
        for layer in self.layers:
            dw1, dw2 = layer.get_gradients()
            w1 = -self.lr * dw1
            w2 = -self.lr * dw2
            layer.update_weights(w1=w1, w2=w2)


class SGDMomentum(Optimizer):
    def __init__(self, layers: list, learning_rate: float, mu: float):
        self.layers = layers
        self.lr = learning_rate
        self.mu = mu
        self.params = self.create_params()

    def create_params(self):
        params = {}
        for layer in self.layers:
            weights = layer.get_weights()
            momentum_params = []
            for w in weights:
                momentum_params.append(np.zeros_like(w))
            params[layer.get_id()] = momentum_params
        return params

    def step(self):
        for layer in self.layers:
            dw1, dw2 = layer.get_gradients()

            self.params[layer.get_id()][0] = self.mu * self.params[layer.get_id()][0] - self.lr * dw1
            self.params[layer.get_id()][1] = self.mu * self.params[layer.get_id()][1] - self.lr * dw2

            layer.update_weights(self.params[layer.get_id()][0], self.params[layer.get_id()][1])


class SGDNesterovMomentum(Optimizer):
    def __init__(self, layers: list, learning_rate: float, mu: float):
        self.layers = layers
        self.lr = learning_rate
        self.mu = mu
        self.params = self.create_params()

    def create_params(self):
        params = {}
        for layer in self.layers:
            weights = layer.get_weights()
            momentum_params = []
            for w in weights:
                momentum_params.append(np.zeros_like(w))
                momentum_params.append(np.zeros_like(w))
            params[layer.get_id()] = momentum_params
        return params

    def step(self):
        for layer in self.layers:
            dw1, dw2 = layer.get_gradients()

            self.params[layer.get_id()][0] = self.params[layer.get_id()][1]
            self.params[layer.get_id()][1] = self.mu * self.params[layer.get_id()][1] - self.lr * dw1

            self.params[layer.get_id()][2] = self.params[layer.get_id()][3]
            self.params[layer.get_id()][3] = self.mu * self.params[layer.get_id()][3] - self.lr * dw2

            w1 = -self.mu * self.params[layer.get_id()][0] + (1.0 + self.mu) * self.params[layer.get_id()][1]
            w2 = -self.mu * self.params[layer.get_id()][2] + (1.0 + self.mu) * self.params[layer.get_id()][3]

            layer.update_weights(w1=w1, w2=w2)
