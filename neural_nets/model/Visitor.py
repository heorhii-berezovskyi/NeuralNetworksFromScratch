from abc import ABCMeta, abstractmethod

from numpy import sum

from neural_nets.model import Relu
from neural_nets.model.BatchNorm import BatchNorm
from neural_nets.model.Linear import Linear

NOT_IMPLEMENTED = "You should implement this."


class Visitor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def visit_linear(self, layer: Linear):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_relu(self, layer: Relu):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_batch_norm(self, layer: BatchNorm):
        raise NotImplementedError(NOT_IMPLEMENTED)


class RegularizationVisitor(Visitor):
    def __init__(self, reg_strength: float):
        self.reg_strength = reg_strength
        self.reg_loss = 0.0

    def visit_linear(self, layer: Linear):
        self.reg_loss += 0.5 * self.reg_strength * sum(layer.W * layer.W)

    def visit_batch_norm(self, layer: BatchNorm):
        pass

    def visit_relu(self, layer: Relu):
        pass

    def reset(self):
        self.reg_loss = 0.0

    def get_reg_loss(self):
        return self.reg_loss


class SGDUpdater(Visitor):
    def __init__(self, reg_strength: float, lr: float):
        self.reg_strength = reg_strength
        self.lr = lr

    def visit_linear(self, layer: Linear):
        layer.dW += self.reg_strength * layer.W

        layer.W -= self.lr * layer.dW
        layer.b -= self.lr * layer.db

    def visit_batch_norm(self, layer: BatchNorm):
        layer.gamma -= self.lr * layer.dgamma
        layer.beta -= self.lr * layer.dbeta

    def visit_relu(self, layer: Relu):
        pass


class SGDMomentumUpdater(Visitor):
    def __init__(self, reg_strength: float, lr: float, mu: float):
        self.reg_strength = reg_strength
        self.lr = lr
        self.mu = mu

    def visit_linear(self, layer: Linear):
        layer.dW += self.reg_strength * layer.W

        layer.v_W = self.mu * layer.v_W - self.lr * layer.dW
        layer.W += layer.v_W

        layer.v_b = self.mu * layer.v_b - self.lr * layer.db
        layer.b += layer.v_b

    def visit_batch_norm(self, layer: BatchNorm):
        layer.v_gamma = self.mu * layer.v_gamma - self.lr * layer.dgamma
        layer.gamma += layer.v_gamma

        layer.v_beta = self.mu * layer.v_beta - self.lr * layer.dbeta
        layer.beta += layer.v_beta

    def visit_relu(self, layer: Relu):
        pass


class SGDNesterovMomentumUpdater(Visitor):
    def __init__(self, reg_strength: float, lr: float, mu: float):
        self.reg_strength = reg_strength
        self.lr = lr
        self.mu = mu

    def visit_linear(self, layer: Linear):
        layer.dW += self.reg_strength * layer.W

        layer.v_W_prev = layer.v_W
        layer.v_W = self.mu * layer.v_W - self.lr * layer.dW
        layer.W += -self.mu * layer.v_W_prev + (1.0 + self.mu) * layer.v_W

        layer.v_b_prev = layer.v_b
        layer.v_b = self.mu * layer.v_b - self.lr * layer.db
        layer.b += -self.mu * layer.v_b_prev + (1.0 + self.mu) * layer.v_b

    def visit_batch_norm(self, layer: BatchNorm):
        layer.v_gamma_prev = layer.v_gamma
        layer.v_gamma = self.mu * layer.v_gamma - self.lr * layer.dgamma
        layer.gamma += -self.mu * layer.v_gamma_prev + (1.0 + self.mu) * layer.v_gamma

        layer.v_beta_prev = layer.v_beta
        layer.v_beta = self.mu * layer.v_beta - self.lr * layer.dbeta
        layer.beta += -self.mu * layer.v_beta_prev + (1.0 + self.mu) * layer.v_beta

    def visit_relu(self, layer: Relu):
        pass


class ModeTuningVisitor(Visitor):
    def __init__(self):
        self.mode = 'train'

    def set_mode(self, mode: str):
        self.mode = mode

    def visit_linear(self, layer: Linear):
        pass

    def visit_batch_norm(self, layer: BatchNorm):
        layer.mode = self.mode

    def visit_relu(self, layer: Relu):
        pass
