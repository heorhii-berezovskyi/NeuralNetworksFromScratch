import numpy as np
from numpy import ndarray

from neural_nets.model.Loss import Loss


class SVM_Loss(Loss):
    def __init__(self, delta: float):
        self.delta = delta
        self.margins = None
        self.labels = None

    def eval_data_loss(self, labels: ndarray, scores: ndarray):
        self.labels = labels
        self.margins = np.maximum(0.0, scores[:, range(labels.size)] - scores[labels, range(labels.size)] + self.delta)
        self.margins[labels, range(labels.size)] = 0.0

        data_loss = self.margins.sum() / labels.size
        return data_loss

    def eval_gradient(self):
        labels = self.labels

        indicators = self.margins
        indicators[indicators > 0.0] = 1.0
        indicators[labels, range(labels.size)] = -indicators[:, range(labels.size)].sum(axis=0)

        dL_dLi = 1.0 / labels.size
        dLi_dscores = indicators

        dL_dscores = dL_dLi * dLi_dscores
        return dL_dscores
