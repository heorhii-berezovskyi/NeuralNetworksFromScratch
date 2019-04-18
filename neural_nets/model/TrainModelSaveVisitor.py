from neural_nets.model.Layer import TrainModeLayer, TrainModeLayerWithWeights
from neural_nets.model.Visitor import TrainLayerVisitor


class TrainModelSaveVisitor(TrainLayerVisitor):
    """
    Initializes a dict of parameters required to build a test model.
    """

    def __init__(self, model_forward_run: list):
        self.model_forward_run = model_forward_run
        self.result = {}

    def visit_weightless_train(self, layer: TrainModeLayer):
        self.model_forward_run.pop()

    def visit_affine_train(self, layer: TrainModeLayerWithWeights):
        self.result = {**self.result.copy(), **layer.content(layer_forward_run=self.model_forward_run.pop())}

    def visit_batch_norm_1d_train(self, layer):
        self.result = {**self.result.copy(), **layer.content(layer_forward_run=self.model_forward_run.pop())}

    def visit_batch_norm_2d_train(self, layer):
        self.result = {**self.result.copy(), **layer.content(layer_forward_run=self.model_forward_run.pop())}

    def get_result(self) -> dict:
        return self.result
