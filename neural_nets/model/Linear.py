from numpy import dot
from numpy import ndarray
from numpy import sum
from numpy import zeros
from numpy.random import randn

from neural_nets.model.Layer import Layer


class Linear(Layer):
    def __init__(self, num_of_neurons: int, input_dim: int):
        self.W = 0.01 * randn(num_of_neurons, input_dim)
        self.b = zeros((num_of_neurons, 1))
        self.input = None
        self.dW = None
        self.db = None

        # Velocity in the Momentum update for W and b.
        self.v_W = zeros((num_of_neurons, input_dim))
        self.v_b = zeros((num_of_neurons, 1))

    def forward(self, input_data: ndarray):
        self.input = input_data
        scores = dot(self.W, input_data) + self.b
        return scores

    def backward(self, dout: ndarray):
        self.dW = dot(dout, self.input.T)
        self.db = sum(dout, axis=1, keepdims=True)
        dinput = dot(self.W.T, dout)
        return dinput

    def get_layer_weights(self):
        return self.W, self.b

    def accept(self, visitor):
        visitor.visit_linear(self)

    def update_with_sgd(self, reg: float, lr: float):
        self.dW += reg * self.W

        self.W -= lr * self.dW
        self.b -= lr * self.db

    def update_with_sgd_momentum(self, reg: float, mu: float, lr: float):
        self.dW += reg * self.W

        self.v_W = mu * self.v_W - lr * self.dW
        self.W += self.v_W

        self.v_b = mu * self.v_b - lr * self.db
        self.b += self.v_b
