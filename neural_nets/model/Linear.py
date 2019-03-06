from numpy import dot
from numpy import empty
from numpy import ndarray
from numpy import sum
from numpy import zeros
from numpy.random import randn

from neural_nets.model.Layer import Layer


class Linear(Layer):
    def __init__(self, num_of_neurons: int, input_dim: tuple):
        self.num_of_neurons = num_of_neurons
        self.scores = empty((num_of_neurons, input_dim[1]))
        self.W = 0.01 * randn(num_of_neurons, input_dim[0])
        self.b = zeros((num_of_neurons, 1))
        self.input = []

    def forward(self, input_data: ndarray):
        self.input.append(input_data)
        self.scores = dot(self.W, input_data) + self.b
        return self.scores

    def backward(self):
        dscores_dinput_data = self.W
        return dscores_dinput_data

    def get_layer_weights(self):
        return self.W, self.b

    def update_layer_weights(self, grad: ndarray, reg: float, lr: float):
        dscores_dW = self.input.pop()
        dW = dot(grad, dscores_dW.T)
        dW += reg * self.W
        db = sum(grad, axis=1, keepdims=True)

        self.W -= lr * dW
        self.b -= lr * db

    def accept(self, visitor):
        visitor.visit_linear(self)
