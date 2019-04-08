import numpy as np

from neural_nets.model.BatchNorm1D import BatchNorm1DTrain
from neural_nets.model.BatchNorm2D import BatchNorm2DTrain
from neural_nets.model.Cache import Cache
from neural_nets.model.Conv2D import Conv2DTrain
from neural_nets.model.Dropout1D import Dropout1DTrain
from neural_nets.model.Dropout2D import Dropout2DTrain
from neural_nets.model.Linear import LinearTrain
from neural_nets.model.MaxPool import MaxPoolTrain
from neural_nets.model.Name import Name
from neural_nets.model.Relu import ReluTrain
from neural_nets.model.Visitor import Visitor


class TestModelInitVisitor(Visitor):
    """
    Initializes a dict of parameters required to build a test model.
    """

    def __init__(self):
        self.result = {}

    def visit_linear(self, layer: LinearTrain):
        pass

    def visit_conv2d(self, layer: Conv2DTrain):
        pass

    def visit_relu(self, layer: ReluTrain):
        pass

    def visit_max_pool(self, layer: MaxPoolTrain):
        pass

    def visit_dropout1d(self, layer: Dropout1DTrain):
        pass

    def visit_dropout2d(self, layer: Dropout2DTrain):
        pass

    def visit_batch_norm_1d(self, layer: BatchNorm1DTrain):
        params = Cache()
        weights = layer.get_weights()

        params.add(name=Name.RUNNING_MEAN, value=np.zeros_like(weights.get(Name.GAMMA)))
        params.add(name=Name.RUNNING_VARIANCE, value=np.zeros_like(weights.get(Name.GAMMA)))

        self.result[layer.get_id()] = params

    def visit_batch_norm_2d(self, layer: BatchNorm2DTrain):
        params = Cache()
        weights = layer.get_weights()

        params.add(name=Name.RUNNING_MEAN, value=np.zeros_like(weights.get(Name.GAMMA)))
        params.add(name=Name.RUNNING_VARIANCE, value=np.zeros_like(weights.get(Name.GAMMA)))

        self.result[layer.get_id()] = params

    def get_result(self):
        return self.result
