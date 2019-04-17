import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TestModeLayer
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TrainLayerVisitor, TestLayerVisitor


class ReluTest(TestModeLayer):
    name = Name.RELU_TEST

    def forward(self, input_data: ndarray) -> ndarray:
        output = np.maximum(0., input_data)
        return output

    def accept(self, visitor: TestLayerVisitor):
        visitor.visit_weightless_test(self)


class ReluTrain(TrainModeLayer):
    name = Name.RELU_TRAIN

    def forward(self, input_data: ndarray, layer_forward_run: Cache) -> Cache:
        output_data = np.maximum(0.0, input_data)
        new_layer_forward_run = Cache()
        new_layer_forward_run.add(name=Name.INPUT, value=input_data)
        new_layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return new_layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache) -> Cache:
        input_data = layer_forward_run.pop(name=Name.INPUT)
        dinput = np.where(input_data > 0., dout, 0.)
        layer_backward_run = Cache()
        layer_backward_run.add(name=Name.D_INPUT, value=dinput)
        return layer_backward_run

    def to_test(self, test_layer_params: Cache) -> TestModeLayer:
        return ReluTest()

    def accept(self, visitor: TrainLayerVisitor):
        visitor.visit_weightless_train(self)
