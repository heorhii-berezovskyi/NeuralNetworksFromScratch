import numpy as np
from numpy import ndarray

from neural_nets.model.Layer import Layer


class Linear(Layer):
    def __init__(self, num_of_neurons: int, input_dim: int):
        self.W = 0.01 * np.random.randn(num_of_neurons, input_dim)
        self.b = np.zeros((num_of_neurons, 1))
        self.input = None
        self.dW = None
        self.db = None

        # Velocity in the Momentum update for W and b.
        self.v_W = np.zeros((num_of_neurons, input_dim))
        self.v_W_prev = np.zeros((num_of_neurons, input_dim))

        self.v_b = np.zeros((num_of_neurons, 1))
        self.v_b_prev = np.zeros((num_of_neurons, 1))

    def forward(self, input_data: ndarray):
        self.input = input_data
        scores = np.dot(self.W, input_data) + self.b
        return scores

    def backward(self, dout: ndarray):
        self.dW = np.dot(dout, self.input.T)
        self.db = np.sum(dout, axis=1, keepdims=True)
        dinput = np.dot(self.W.T, dout)
        return dinput

    def get_layer_weights(self):
        return self.W, self.b

    def accept(self, visitor):
        visitor.visit_linear(self)

