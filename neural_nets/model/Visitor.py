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

    def get_reg_loss(self):
        return self.reg_loss


class SGDUpdater(Visitor):
    def __init__(self, reg_strength: float, lr: float):
        self.reg_strength = reg_strength
        self.lr = lr

    def visit_linear(self, layer: Linear):
        layer.update_with_sgd(reg=self.reg_strength, lr=self.lr)

    def visit_batch_norm(self, layer: BatchNorm):
        layer.update_with_sgd(lr=self.lr)

    def visit_relu(self, layer: Relu):
        pass


class SGDMomentumUpdater(Visitor):
    def __init__(self, reg_strength: float, lr: float, mu: float):
        self.reg_strength = reg_strength
        self.lr = lr
        self.mu = mu

    def visit_linear(self, layer: Linear):
        layer.update_with_sgd_momentum(reg=self.reg_strength, mu=self.mu, lr=self.lr)

    def visit_batch_norm(self, layer: BatchNorm):
        layer.update_with_sgd_momentum(mu=self.mu, lr=self.lr)

    def visit_relu(self, layer: Relu):
        pass


class TestTimeRunner(Visitor):

    def visit_linear(self, layer: Linear):
        pass

    def visit_batch_norm(self, layer: BatchNorm):
        layer.mode = 'test'

    def visit_relu(self, layer: Relu):
        pass
