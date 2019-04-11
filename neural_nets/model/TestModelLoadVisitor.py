from numpy.lib.npyio import NpzFile

from neural_nets.model.Layer import TestModeLayer, TestModeLayerWithWeights, TestModeLayerWithWeightsAndParams
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TestLayerBaseVisitor


class TestModelLoadVisitor(TestLayerBaseVisitor):
    """
    Initializes a dict of parameters required to build a test model.
    """

    def __init__(self, all_params: NpzFile):
        self.all_params = all_params

    def visit_weightless_test(self, layer: TestModeLayer):
        pass

    def visit_affine_test(self, layer: TestModeLayerWithWeights):
        weight_names = [Name.WEIGHTS, Name.BIASES]
        self._load_weights(layer=layer, weight_names=weight_names)

    def visit_batch_norm_test(self, layer: TestModeLayerWithWeightsAndParams):
        weight_names = [Name.GAMMA, Name.BETA]
        self._load_weights(layer=layer, weight_names=weight_names)

        layer_name_plus_id = layer.get_name().value + str(layer.get_id())
        for param_name in [Name.RUNNING_MEAN, Name.RUNNING_VARIANCE]:
            param_key = layer_name_plus_id + param_name.value
            param_value = self.all_params[param_key]
            layer.get_params().update(name=param_name, value=param_value)

    def _load_weights(self, layer: TestModeLayerWithWeights, weight_names: list):
        layer_name_plus_id = layer.get_name().value + str(layer.get_id())
        for weight_name in weight_names:
            weight_key = layer_name_plus_id + weight_name.value
            weight_value = self.all_params[weight_key]
            layer.get_weights().update(name=weight_name, value=weight_value)
