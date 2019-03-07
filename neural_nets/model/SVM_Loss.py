from numpy import amax
from numpy import empty
from numpy import exp
from numpy import log
from numpy import sum
from numpy import maximum

from neural_nets.model.Layer import Layer


class SVM_Loss(Layer):
    def __init__(self, input_dim: tuple, delta: float):
        self.delta = delta
        self.margins = empty(input_dim)
        self.input = []

    def forward(self, labels_and_scores: tuple):
        self.input.append(labels_and_scores)
        labels, scores = labels_and_scores

        self.margins = maximum(0.0, scores[:, range(labels.size)] - scores[labels, range(labels.size)] + self.delta)
        self.margins[labels, range(labels.size)] = 0.0

        data_loss = self.margins.sum() / labels.size
        return data_loss

    def backward(self):
        labels, scores = self.input.pop()

        indicators = self.margins
        indicators[indicators > 0.0] = 1.0
        indicators[labels, range(labels.size)] = -indicators[:, range(labels.size)].sum(axis=0)

        dL_dLi = 1.0 / labels.size
        dLi_dscores = indicators

        dL_dscores = dL_dLi * dLi_dscores
        return dL_dscores

    def accept(self, visitor):
        visitor.visit_svm_loss(self)
