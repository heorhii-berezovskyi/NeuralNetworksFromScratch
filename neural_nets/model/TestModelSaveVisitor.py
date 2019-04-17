from neural_nets.model.Layer import TestModeLayer, TestModeLayerWithWeights
from neural_nets.model.Visitor import TestLayerVisitor


class TestModelSaveVisitor(TestLayerVisitor):
    """
    Initializes a dict of parameters required to build a test model.
    """

    def __init__(self):
        self.result = {}

    def visit_weightless_test(self, layer: TestModeLayer):
        pass

    def visit_weighted_test(self, layer: TestModeLayerWithWeights):
        self.result = {**self.result.copy(), **layer.content()}

    def get_result(self) -> dict:
        return self.result
