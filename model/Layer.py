from numpy import amax
from numpy import exp
from numpy import sum
from numpy import log
from numpy import empty
from numpy import full_like
from numpy import dot
from numpy import ndarray
from numpy.random import randn
from numpy import zeros
from numpy import maximum


class Layer(object):
    def __init__(self, batch_dim: tuple):
        self.batch_dim = batch_dim

    def forward(self, input_data: tuple):
        pass

    def backward(self, input_data: tuple):
        pass


class Linear(Layer):
    def __init__(self, num_of_neurons: int, batch_dim: tuple, reg: float):
        super(Linear, self).__init__(batch_dim)
        self.reg = reg
        self.num_of_neurons = num_of_neurons
        self.scores = empty((num_of_neurons, batch_dim[1]))
        self.W = 0.01 * randn(num_of_neurons, batch_dim[0])
        self.b = zeros((num_of_neurons, 1))

    def forward(self, input_data: ndarray):
        self.scores = dot(self.W, input_data) + self.b
        return self.scores

    def backward(self, input_data: ndarray):
        dscores_dW = input_data
        dscores_dW += self.reg * self.W
        dscores_db = full_like(self.b, 1.0)
        return dscores_dW, dscores_db


class Relu(Layer):
    def __init__(self, batch_dim: tuple):
        super(Relu, self).__init__(batch_dim)

    def forward(self, input_data: ndarray):
        return maximum(0.0, input_data)

    def backward(self, input_data: ndarray):
        drelu_dscores = full_like(input_data, 1.0)
        drelu_dscores[input_data <= 0] = 0.0
        return drelu_dscores


class CrossEntropyLoss(Layer):
    def __init__(self, batch_dim: tuple):
        super(CrossEntropyLoss, self).__init__(batch_dim)
        self.probs = empty(batch_dim)

    def forward(self, labels_and_scores: tuple):
        labels, scores = labels_and_scores

        # Subtracting min values from scores for numeric stability.
        scores -= amax(scores, axis=0)

        exp_scores = exp(scores)
        # Calculating probabilities for each class over a mini-batch.
        self.probs = exp_scores / sum(exp_scores, axis=0, keepdims=True)

        # Losses of each image.
        correct_logprobs = -log(self.probs[labels, range(labels.size)])

        # Loss over a mini-batch.
        loss = sum(correct_logprobs) / labels.size
        return loss

    def backward(self, labels_and_scores: tuple):
        labels, scores = labels_and_scores
        dL_dLi = 1.0 / labels.size

        # dLi_scores = probs[k] - 1(yi = k)
        dLi_dscores = self.probs
        dLi_dscores[labels, range(labels.size)] -= 1.0

        dL_dscores = dL_dLi * dLi_dscores
        return dL_dscores
