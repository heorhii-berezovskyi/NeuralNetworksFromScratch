from numpy import mean
from numpy import ndarray
from numpy import sqrt
from numpy import sum
from numpy import var
from numpy import zeros
from numpy.random import randn

from neural_nets.model.Layer import Layer


class BatchNorm(Layer):
    def __init__(self, input_dim: int, momentum: float):

        self.mode = 'train'
        self.gamma = 0.01 * randn(input_dim, 1)
        self.beta = 0.01 * randn(input_dim, 1)

        self.input = None
        self.mu = None
        self.var = None
        self.X_norm = None

        self.dgamma = None
        self.dbeta = None

        self.running_mean = zeros((input_dim, 1))
        self.running_var = zeros((input_dim, 1))
        self.momentum = momentum

        # Velocity in the Momentum update for gamma and beta.
        self.v_gamma = zeros((input_dim, 1))
        self.v_beta = zeros((input_dim, 1))

    def forward(self, input_data: ndarray):
        if self.mode == 'train':
            self.input = input_data

            self.mu = mean(input_data, axis=1, keepdims=True)
            self.var = var(input_data, axis=1, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            self.X_norm = (input_data - self.mu) / sqrt(self.var + 1e-5)
            out = self.gamma * self.X_norm + self.beta

        elif self.mode == 'test':
            out = (input_data - self.running_mean) / sqrt(self.running_var + 1e-5)
            out = self.gamma * out + self.beta
            self.mode = 'train'
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % self.mode)
        return out

    def backward(self, dout: ndarray):
        input_data = self.input
        N = input_data.shape[1]

        X_mu = input_data - self.mu
        std_inv = 1. / sqrt(self.var + 1e-5)

        dX_norm = dout * self.gamma
        dvar = sum(dX_norm * X_mu, axis=1, keepdims=True) * -.5 * std_inv ** 3

        dmu = sum(dX_norm * -std_inv, axis=1, keepdims=True) + dvar * mean(-2. * X_mu, axis=1, keepdims=True)

        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)

        self.dgamma = sum(dout * self.X_norm, axis=1, keepdims=True)
        self.dbeta = sum(dout, axis=1, keepdims=True)

        return dX

    def get_layer_weights(self):
        return self.gamma, self.beta

    def accept(self, visitor):
        visitor.visit_batch_norm(self)

    def update_with_sgd(self, lr: float):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta

    def update_with_sgd_momentum(self, mu: float, lr: float):
        self.v_gamma = mu * self.v_gamma - lr * self.dgamma
        self.gamma += self.v_gamma

        self.v_beta = mu * self.v_beta - lr * self.dbeta
        self.beta += self.v_beta
