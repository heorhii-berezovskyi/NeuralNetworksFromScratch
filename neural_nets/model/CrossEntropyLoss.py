import numpy as np
from numpy import ndarray

from neural_nets.model.Loss import Loss


class CrossEntropyLoss(Loss):
    def __init__(self):
        self.labels = None
        self.probs = None

    def eval_data_loss(self, labels: ndarray, scores: ndarray):
        self.labels = labels
        # Subtracting min values from scores for numeric stability.
        scores -= np.amax(scores, axis=0)

        exp_scores = np.exp(scores)
        # Calculating probabilities for each class over a mini-batch.
        self.probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

        # Losses of each image.
        correct_logprobs = -np.log(self.probs[labels, range(labels.size)])

        # Loss over a mini-batch.
        data_loss = np.sum(correct_logprobs) / labels.size
        return data_loss

    def eval_gradient(self):
        labels = self.labels
        dL_dLi = 1.0 / labels.size

        # dLi_scores = probs[k] - 1(yi = k)
        dLi_dscores = self.probs
        dLi_dscores[labels, range(labels.size)] -= 1.0

        dL_dscores = dL_dLi * dLi_dscores
        return dL_dscores
