from abc import ABCMeta, abstractmethod
from neural_nets.model.Visitor import SGDUpdater, SGDMomentumUpdater

NOT_IMPLEMENTED = "You should implement this."


class Optimizer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(self):
        raise NotImplementedError(NOT_IMPLEMENTED)


class SGD(Optimizer):
    def __init__(self, layers: list, learning_rate: float, reg: float):
        self.layers = layers
        self.learning_rate = learning_rate
        self.visitor = SGDUpdater(reg_strength=reg, lr=learning_rate)

    def step(self):
        for layer in self.layers:
            layer.accept(self.visitor)


class SGDMomentum(Optimizer):
    def __init__(self, layers: list, learning_rate: float, reg: float, mu: float):
        self.layers = layers
        self.learning_rate = learning_rate
        self.reg = reg
        self.mu = mu
        self.visitor = SGDMomentumUpdater(reg_strength=reg, lr=learning_rate, mu=mu)

    def step(self):
        for layer in self.layers:
            layer.accept(self.visitor)
