from numpy import amax
from numpy import exp
from numpy import log
from numpy import ndarray
from numpy import sum

from neural_nets.model.Layer import Layer


class CrossEntropyLoss(Layer):
    def __init__(self):
        self.probs = None
        self.input = None

    def forward(self, labels_and_scores: tuple):
        self.input = labels_and_scores
        labels, scores = labels_and_scores

        # Subtracting min values from scores for numeric stability.
        scores -= amax(scores, axis=0)

        exp_scores = exp(scores)
        # Calculating probabilities for each class over a mini-batch.
        probs = exp_scores / sum(exp_scores, axis=0, keepdims=True)
        self.probs = probs

        # Losses of each image.
        correct_logprobs = -log(probs[labels, range(labels.size)])

        # Loss over a mini-batch.
        data_loss = sum(correct_logprobs) / labels.size
        return data_loss

    def backward(self, dout: ndarray):
        labels, scores = self.input
        dL_dLi = 1.0 / labels.size

        # dLi_scores = probs[k] - 1(yi = k)
        dLi_dscores = self.probs
        dLi_dscores[labels, range(labels.size)] -= 1.0

        dL_dscores = dL_dLi * dLi_dscores
        return dL_dscores
