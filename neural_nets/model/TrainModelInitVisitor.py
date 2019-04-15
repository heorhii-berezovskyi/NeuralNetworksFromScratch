import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TrainModeLayerWithWeights
from neural_nets.model.Name import Name
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

    def visit_batch_norm_train(self, layer: TrainModeLayerWithWeights):
        weights = layer.get_weights()

        params = Cache()
        params.add(name=Name.RUNNING_MEAN, value=np.zeros_like(weights.get(name=Name.GAMMA), dtype=np.float64))
        params.add(name=Name.RUNNING_VAR, value=np.zeros_like(weights.get(name=Name.GAMMA), dtype=np.float64))
        self.result.append(params)

    def get_result(self) -> list:
        return self.result
