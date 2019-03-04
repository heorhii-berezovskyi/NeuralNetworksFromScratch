from numpy import amax
from numpy import argmax
from numpy import dot
from numpy import exp
from numpy import log
from numpy import maximum
from numpy import mean
from numpy import multiply
from numpy import ndarray
from numpy import power
from numpy import sqrt
from numpy import sum
from numpy import zeros
from numpy.random import randn


class TwoLayerNNCrossEntropyBatchNorm:
    def __init__(self, batch_size: int, num_of_classes: int, size_of_input_vector: int, num_of_hidden_neurons: int,
                 reg: float):
        self.batch_size = batch_size
        self.num_of_hidden_neurons = num_of_hidden_neurons
        self.W = 0.01 * randn(num_of_hidden_neurons, size_of_input_vector)
        self.b = zeros((num_of_hidden_neurons, 1))

        self.W2 = 0.01 * randn(num_of_classes, num_of_hidden_neurons)
        self.b2 = zeros((num_of_classes, 1))

        self.gamma = 0.01 * randn(num_of_hidden_neurons, 1)
        self.beta = 0.01 * randn(num_of_hidden_neurons, 1)

        self.reg = reg

    def eval_batch_normalized_transform(self, activations: ndarray):
        mini_batch_mean = mean(activations, axis=1).reshape(self.num_of_hidden_neurons, 1)
        mini_batch_variance = mean(power(activations - mini_batch_mean, 2), axis=1).reshape(self.num_of_hidden_neurons,
                                                                                            1)
        normalized_activations = (activations - mini_batch_mean) / sqrt(mini_batch_variance + 1e-10)
        scaled_and_shifted = multiply(normalized_activations, self.gamma) + self.beta
        return scaled_and_shifted

    def eval_loss(self, labels: ndarray, images: ndarray):
        activations = maximum(0.0, dot(self.W, images.T) + self.b)
        scaled_and_shifted = self.eval_batch_normalized_transform(activations)
        scores = dot(self.W2, scaled_and_shifted) + self.b2
        scores -= amax(scores, axis=0)

        exp_scores = exp(scores)
        probs = exp_scores / sum(exp_scores, axis=0, keepdims=True)

        correct_log_pros = -log(probs[labels, range(self.batch_size)])
        data_loss = sum(correct_log_pros) / self.batch_size
        reg_loss = 0.5 * self.reg * sum(self.W * self.W) + 0.5 * self.reg * sum(self.W2 * self.W2)
        loss = data_loss + reg_loss
        return loss

    def eval_gradient(self, labels: ndarray, images: ndarray):
        activations = maximum(0.0, dot(self.W, images.T) + self.b)
        mini_batch_mean = mean(activations, axis=1).reshape(self.num_of_hidden_neurons, 1)
        mini_batch_variance = mean(power(activations - mini_batch_mean, 2), axis=1).reshape(self.num_of_hidden_neurons,
                                                                                            1)
        normalized_activations = (activations - mini_batch_mean) / sqrt(mini_batch_variance + 1e-10)
        scaled_and_shifted = multiply(normalized_activations, self.gamma) + self.beta

        scores = dot(self.W2, scaled_and_shifted) + self.b2
        scores -= amax(scores, axis=0)

        exp_scores = exp(scores)
        probs = exp_scores / sum(exp_scores, axis=0, keepdims=True)

        dscores = probs
        dscores[labels, range(self.batch_size)] -= 1.0
        dscores /= self.batch_size

        dW2 = dot(dscores, activations.T)
        db2 = sum(dscores, axis=1, keepdims=True)

        dscaled_and_shifted = dot(self.W2.T, dscores)

        dnormalized_activations = multiply(dscaled_and_shifted, self.gamma)

        dmini_batch_variance = multiply(dnormalized_activations * (activations - mini_batch_mean), -0.5 * power(
            mini_batch_variance + 1e-10, -3.0 / 2.0))

        dmini_batch_mean = multiply(dnormalized_activations, -1.0 / sqrt(mini_batch_variance + 1e-10))

        dactivations = multiply(dnormalized_activations, 1.0 / sqrt(mini_batch_variance + 1e-10)) + multiply(
            -2.0 * (activations - mini_batch_mean) / self.batch_size,
            dmini_batch_variance) + dmini_batch_mean / self.batch_size
        dactivations[activations <= 0.0] = 0.0

        dgamma = sum(multiply(dscaled_and_shifted, normalized_activations), axis=1).reshape(self.num_of_hidden_neurons,
                                                                                            1)
        dbeta = sum(dscaled_and_shifted, axis=1).reshape(self.num_of_hidden_neurons, 1)

        dW = dot(dactivations, images)
        db = sum(dactivations, axis=1, keepdims=True)

        dW2 += self.reg * self.W2
        dW += self.reg * self.W
        return dW, db, dW2, db2, dgamma, dbeta

    def train_iter(self, labels: ndarray, images: ndarray, lr: float):
        dW, db, dW2, db2, dgamma, dbeta = self.eval_gradient(labels, images)

        self.W -= lr * dW
        self.b -= lr * db
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta

    def test(self, labels: ndarray, images: ndarray):
        activations = maximum(0.0, dot(self.W, images.T) + self.b)
        scaled_and_shifted = self.eval_batch_normalized_transform(activations)
        scores = dot(self.W2, scaled_and_shifted) + self.b2
        predicted_class = argmax(scores, axis=0)
        accuracy = mean(predicted_class == labels)
        return accuracy
