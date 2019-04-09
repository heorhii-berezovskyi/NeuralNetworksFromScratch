import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TrainModeLayerWithWeights
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import Visitor


class TestModelInitVisitor(Visitor):
    """
    Initializes a dict of parameters required to build a test model.
    """

    def __init__(self):
        self.result = {}

    def visit_linear(self, layer: TrainModeLayerWithWeights):
        pass

    def visit_weightless_layer(self, layer: TrainModeLayer):
        pass

    def visit_batch_norm(self, layer: TrainModeLayerWithWeights):
        params = Cache()
        weights = layer.get_weights()

        params.add(name=Name.RUNNING_MEAN, value=np.zeros_like(weights.get(name=Name.GAMMA), dtype=np.float64))
        params.add(name=Name.RUNNING_VARIANCE, value=np.zeros_like(weights.get(name=Name.GAMMA), dtype=np.float64))

        self.result[layer.get_id()] = params

    def get_result(self) -> dict:
        return self.result
