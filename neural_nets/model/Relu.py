import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TestModeLayer
from neural_nets.model.Name import Name


class ReluTest(TestModeLayer):
    def __init__(self):
        super().__init__()

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.RELU_TEST

    def forward(self, input_data: ndarray):
        output = np.maximum(0., input_data)
        return output


class ReluTrain(TrainModeLayer):
    def __init__(self):
        super().__init__()

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.RELU_TRAIN

    def forward(self, input_data: ndarray, test_model_params: dict):
        output_data = np.maximum(0.0, input_data)
        layer_forward_run = Cache()
        layer_forward_run.add(name=Name.INPUT, value=input_data)
        layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache):
        input_data = layer_forward_run.get(name=Name.INPUT)
        dinput = np.where(input_data > 0.0, dout, 0.0)
        layer_backward_run = Cache()
        return dinput, layer_backward_run

    def to_test(self, test_model_params: dict):
        return ReluTest()

    def accept(self, visitor):
        # visitor.visit_relu(self)
        visitor.visit_weightless_layer(self)
