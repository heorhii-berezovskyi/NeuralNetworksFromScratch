from neural_nets.model.BatchNorm1D import BatchNorm1DTrain
from neural_nets.model.BatchNorm2D import BatchNorm2DTrain
from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TrainModeLayerWithWeights
from neural_nets.model.Visitor import TrainLayerVisitor


class TrainModelInitVisitor(TrainLayerVisitor):
    """
    Initializes a list of parameters required to build a test model.
    """

    def __init__(self):
        self.result = []

    def visit_affine_train(self, layer: TrainModeLayerWithWeights):
        self.result.append(Cache())

    def visit_weightless_train(self, layer: TrainModeLayer):
        self.result.append(Cache())

    def visit_batch_norm_1d_train(self, layer: BatchNorm1DTrain):
        self.result.append(layer.init_params())

    def visit_batch_norm_2d_train(self, layer: BatchNorm2DTrain):
        self.result.append(layer.init_params())

    def get_result(self) -> list:
        return self.result
