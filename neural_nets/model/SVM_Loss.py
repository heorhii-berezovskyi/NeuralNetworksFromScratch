from numpy import maximum
from numpy import ndarray

from neural_nets.model.Layer import Layer


class SVM_Loss(Layer):
    def __init__(self, delta: float):
        self.delta = delta
        self.margins = None
        self.input = None

    def forward(self, labels_and_scores: tuple):
        self.input = labels_and_scores
        labels, scores = labels_and_scores

        self.margins = maximum(0.0, scores[:, range(labels.size)] - scores[labels, range(labels.size)] + self.delta)
        self.margins[labels, range(labels.size)] = 0.0

        data_loss = self.margins.sum() / labels.size
        return data_loss

    def backward(self, dout: ndarray):
        labels, scores = self.input

        indicators = self.margins
        indicators[indicators > 0.0] = 1.0
        indicators[labels, range(labels.size)] = -indicators[:, range(labels.size)].sum(axis=0)

        dL_dLi = 1.0 / labels.size
        dLi_dscores = indicators

        dL_dscores = dL_dLi * dLi_dscores
        return dL_dscores
