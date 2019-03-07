from numpy import amax
from numpy import empty
from numpy import exp
from numpy import log
from numpy import sum

from neural_nets.model.Layer import Layer


class CrossEntropyLoss(Layer):
    def __init__(self, input_dim: tuple):
        self.probs = empty(input_dim)
        self.input = []

    def forward(self, labels_and_scores: tuple):
        self.input.append(labels_and_scores)
        labels, scores = labels_and_scores

        # Subtracting min values from scores for numeric stability.
        scores -= amax(scores, axis=0)

        exp_scores = exp(scores)
        # Calculating probabilities for each class over a mini-batch.
        self.probs = exp_scores / sum(exp_scores, axis=0, keepdims=True)

        # Losses of each image.
        correct_logprobs = -log(self.probs[labels, range(labels.size)])

        # Loss over a mini-batch.
        data_loss = sum(correct_logprobs) / labels.size
        return data_loss

    def backward(self):
        labels, scores = self.input.pop()
        dL_dLi = 1.0 / labels.size

        # dLi_scores = probs[k] - 1(yi = k)
        dLi_dscores = self.probs
        dLi_dscores[labels, range(labels.size)] -= 1.0

        dL_dscores = dL_dLi * dLi_dscores
        return dL_dscores

    def accept(self, visitor):
        visitor.visit_cross_entropy_loss(self)
