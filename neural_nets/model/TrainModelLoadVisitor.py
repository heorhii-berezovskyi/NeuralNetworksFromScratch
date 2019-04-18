from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TrainModeLayerWithWeights
from neural_nets.model.Visitor import TrainLayerVisitor


class TrainModelLoadVisitor(TrainLayerVisitor):
    """
    Initializes a dict of parameters required to build a test model.
    """

    def __init__(self, all_params):
        self.all_params = all_params
        self.layers = []
        self.model_forward_run = []

    def visit_weightless_train(self, layer: TrainModeLayer):
        self.layers.append(layer)
        self.model_forward_run.append(Cache())

    def visit_affine_train(self, layer: TrainModeLayerWithWeights):
        layer, layer_forward_run = layer.from_params(self.all_params)
        self.layers.append(layer)
        self.model_forward_run.append(layer_forward_run)

    def visit_batch_norm_1d_train(self, layer):
        layer, layer_forward_run = layer.from_params(self.all_params)
        self.layers.append(layer)
        self.model_forward_run.append(layer_forward_run)

    def visit_batch_norm_2d_train(self, layer):
        layer, layer_forward_run = layer.from_params(self.all_params)
        self.layers.append(layer)
        self.model_forward_run.append(layer_forward_run)

    def get_result(self) -> tuple:
        return self.layers, self.model_forward_run
