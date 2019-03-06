from numpy import empty
from numpy import maximum
from numpy import ndarray

from neural_nets.model.Layer import Layer


class Relu(Layer):
    def __init__(self, input_dim: tuple):
        self.scores = empty(input_dim)
        self.input = []

    def forward(self, input_data: ndarray):
        self.input.append(input_data)
        self.scores = maximum(0.0, input_data)
        return self.scores

    def backward(self):
        input_data = self.input.pop()
        drelu_input_data = self.scores
        drelu_input_data[input_data > 0.0] = 1.0
        return drelu_input_data

    def accept(self, visitor):
        visitor.visit_relu(self)
