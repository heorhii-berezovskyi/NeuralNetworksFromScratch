from numpy import maximum
from numpy import ndarray
from numpy import multiply

from neural_nets.model.Layer import Layer


class Relu(Layer):
    def __init__(self):
        self.act = None
        self.input = None

    def forward(self, input_data: ndarray):
        self.input = input_data
        self.act = maximum(0.0, input_data)
        return self.act

    def backward(self, dout: ndarray):
        input_data = self.input
        dact_input_data = self.act
        dact_input_data[input_data > 0.0] = 1.0
        dinput_data = multiply(dout, dact_input_data)
        return dinput_data

    def accept(self, visitor):
        visitor.visit_relu(self)
