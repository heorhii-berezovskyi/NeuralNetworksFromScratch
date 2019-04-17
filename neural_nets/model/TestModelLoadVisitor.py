from neural_nets.model.Layer import TestModeLayer, TestModeLayerWithWeights
from neural_nets.model.Visitor import TestLayerVisitor


class TestModelLoadVisitor(TestLayerVisitor):
    """
    Initializes a dict of parameters required to build a test model.
    """

    def __init__(self, all_params):
        self.all_params = all_params
        self.result = []

    def visit_weighted_test(self, layer: TestModeLayerWithWeights):
        self.result.append(layer.from_params(self.all_params))

    def visit_weightless_test(self, layer: TestModeLayer):
        self.result.append(layer)

    def get_result(self) -> list:
        return self.result
