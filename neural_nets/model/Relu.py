import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TestModeLayer
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TrainLayerBaseVisitor, TestLayerBaseVisitor


class ReluTest(TestModeLayer):
    def __init__(self, layer_id: int):
        self.id = layer_id

    def get_id(self) -> int:
        return self.id

    def get_name(self) -> Name:
        return Name.RELU_TEST

    def forward(self, input_data: ndarray) -> ndarray:
        output = np.maximum(0., input_data)
        return output

    def accept(self, visitor: TestLayerBaseVisitor):
        visitor.visit_weightless_test(self)


class ReluTrain(TrainModeLayer):
    def __init__(self):
        super().__init__()

    def get_id(self) -> int:
        return self.id

    def get_name(self) -> Name:
        return Name.RELU_TRAIN

    def forward(self, input_data: ndarray, test_model_params: dict) -> Cache:
        output_data = np.maximum(0.0, input_data)
        layer_forward_run = Cache()
        layer_forward_run.add(name=Name.INPUT, value=input_data)
        layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache) -> tuple:
        input_data = layer_forward_run.get(name=Name.INPUT)
        dinput = np.where(input_data > 0., dout, 0.)
        layer_backward_run = Cache()
        return dinput, layer_backward_run

    def to_test(self, test_model_params: dict) -> TestModeLayer:
        return ReluTest(layer_id=self.id)

    def accept(self, visitor: TrainLayerBaseVisitor):
        visitor.visit_weightless_train(self)
