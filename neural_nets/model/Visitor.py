from abc import ABCMeta, abstractmethod

from numpy import dot
from numpy import multiply
from numpy import ndarray
from numpy import sum

from neural_nets.model import Relu
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.model.Linear import Linear
from neural_nets.model.SVM_Loss import SVM_Loss

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
    def visit_cross_entropy_loss(self, layer: CrossEntropyLoss):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def visit_svm_loss(self, layer: SVM_Loss):
        raise NotImplementedError(NOT_IMPLEMENTED)


class RegularizationVisitor(Visitor):
    def __init__(self, reg_strength: float):
        self.reg_strength = reg_strength
        self.reg_loss = 0.0

    def visit_linear(self, layer: Linear):
        self.reg_loss += 0.5 * self.reg_strength * sum(layer.W * layer.W)

    def visit_relu(self, layer: Relu):
        pass

    def visit_cross_entropy_loss(self, layer: CrossEntropyLoss):
        pass

    def visit_svm_loss(self, layer: SVM_Loss):
        pass

    def get_reg_loss(self):
        return self.reg_loss


class GradientUpdateVisitor(Visitor):
    def __init__(self, loss_grad: ndarray, updating_loss_grad_term: ndarray, reg_strength: float, lr: float):
        self.loss_grad = loss_grad
        self.updating_loss_grad_term = updating_loss_grad_term
        self.reg_strength = reg_strength
        self.lr = lr
        self.new_loss_grad = []

    def visit_linear(self, layer: Linear):
        layer.update_layer_weights(grad=self.loss_grad, reg=self.reg_strength, lr=self.lr)
        self.new_loss_grad.append(dot(self.updating_loss_grad_term.T, self.loss_grad))

    def visit_relu(self, layer: Relu):
        self.new_loss_grad.append(multiply(self.loss_grad, self.updating_loss_grad_term))

    def visit_cross_entropy_loss(self, layer: CrossEntropyLoss):
        pass

    def visit_svm_loss(self, layer: SVM_Loss):
        pass

    def get_updated_loss_gradient(self):
        return self.new_loss_grad.pop()
