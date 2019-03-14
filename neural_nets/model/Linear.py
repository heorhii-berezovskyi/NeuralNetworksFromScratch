import numpy as np
from numpy import ndarray

from neural_nets.model.Layer import LayerWithWeights


class Linear(LayerWithWeights):
    def __init__(self, num_of_neurons: int, input_dim: int):
        super().__init__()
        self.W = 0.01 * np.random.randn(num_of_neurons, input_dim)
        self.b = np.zeros((num_of_neurons, 1))
        self.input = None
        self.dW = None
        self.db = None

        self.mode = 'train'

    def forward(self, input_data: ndarray):
        self.input = input_data
        scores = np.dot(self.W, input_data) + self.b
        return scores

    def backward(self, dout: ndarray):
        self.dW = np.dot(dout, self.input.T)
        self.db = np.sum(dout, axis=1, keepdims=True)
        dinput = np.dot(self.W.T, dout)
        return dinput

    def set_layer_mode(self, mode: str):
        self.mode = mode

    def get_weights(self):
        return self.W, self.b

    def get_gradients(self):
        return self.dW, self.db

    def update_weights(self, w1: ndarray, w2: ndarray):
        self.W += w1
        self.b += w2

    def accept(self, visitor):
        visitor.visit_layer_with_weights(self)
