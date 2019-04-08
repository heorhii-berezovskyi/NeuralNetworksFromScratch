import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TestModeLayer
from neural_nets.model.Name import Name


class Dropout2DTest(TestModeLayer):
    def __init__(self):
        super().__init__()

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.DROPOUT2D_TEST

    def forward(self, input_data: ndarray):
        output_data = input_data
        return output_data


class Dropout2DTrain(TrainModeLayer):
    def __init__(self, keep_active_prob: float):
        super().__init__()
        self.p = keep_active_prob

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.DROPOUT2D_TRAIN

    def forward(self, input_data: ndarray, test_model_params: dict):
        N, C = input_data.shape[0], input_data.shape[1]
        mask = (np.random.rand(N, C, 1, 1) < self.p) / self.p
        output_data = input_data * mask

        layer_forward_run = Cache()
        layer_forward_run.add(name=Name.MASK, value=mask)
        layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache):
        mask = layer_forward_run.get(name=Name.MASK)
        dinput = dout * mask

        layer_backward_run = Cache()
        return dinput, layer_backward_run

    def to_test(self, test_model_params: dict):
        layer = Dropout2DTest()
        return layer

    def accept(self, visitor):
        visitor.visit_dropout2d(self)
