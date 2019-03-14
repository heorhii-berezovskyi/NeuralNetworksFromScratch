import numpy as np
from numpy import ndarray

from neural_nets.model.Layer import Layer


class Relu(Layer):
    def __init__(self):
        self.activations = None
        self.input_data = None

    def forward(self, input_data: ndarray):
        self.input_data = input_data
        self.activations = np.maximum(0.0, input_data)
        return self.activations

    def backward(self, dout: ndarray):
        input_data = self.input_data
        dact_input_data = self.activations
        dact_input_data[input_data > 0.0] = 1.0
        dinput_data = np.multiply(dout, dact_input_data)
        return dinput_data

    def accept(self, visitor):
        visitor.visit_layer(self)
