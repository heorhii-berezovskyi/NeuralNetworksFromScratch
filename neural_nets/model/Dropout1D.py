import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TestModeLayer
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TestLayerVisitor, TrainLayerVisitor


class Dropout1DTest(TestModeLayer):
    name = Name.DROPOUT1D_TEST

    def __init__(self, layer_id: int):
        self.id = layer_id

    def forward(self, input_data: ndarray) -> ndarray:
        output_data = input_data
        return output_data

    def accept(self, visitor: TestLayerVisitor):
        visitor.visit_weightless_test(self)


class Dropout1DTrain(TrainModeLayer):
    name = Name.DROPOUT1D_TRAIN

    def __init__(self, layer_id: int, keep_active_prob: float):
        self.id = layer_id
        self.p = keep_active_prob

    def forward(self, input_data: ndarray, layer_forward_run: Cache) -> Cache:
        mask = (np.random.rand(*input_data.shape) < self.p) / self.p
        output_data = input_data * mask

        new_layer_forward_run = Cache()
        new_layer_forward_run.add(name=Name.MASK, value=mask)
        new_layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return new_layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache) -> Cache:
        dinput = dout * layer_forward_run.pop(name=Name.MASK)

        layer_backward_run = Cache()
        layer_backward_run.add(name=Name.D_INPUT, value=dinput)
        return layer_backward_run

    def to_test(self, layer_forward_run: Cache) -> TestModeLayer:
        return Dropout1DTest(layer_id=self.id)

    def accept(self, visitor: TrainLayerVisitor):
        visitor.visit_weightless_train(self)
