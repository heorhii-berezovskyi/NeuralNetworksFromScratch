from collections import OrderedDict

from neural_nets.model.Layer import TestModeLayer, TestModeLayerWithWeights, TestModeLayerWithWeightsAndParams
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TestLayerBaseVisitor


class TestModelSaveVisitor(TestLayerBaseVisitor):
    """
    Initializes a dict of parameters required to build a test model.
    """

    def __init__(self):
        self.result = OrderedDict()

    def visit_weightless_test(self, layer: TestModeLayer):
        pass

    def visit_affine_test(self, layer: TestModeLayerWithWeights):
        layer_name_plus_id = layer.get_name().value + str(layer.get_id())
        layer_weights = layer.get_weights()

        weights = layer_weights.get(name=Name.WEIGHTS)
        weights_key = layer_name_plus_id + Name.WEIGHTS.value

        biases = layer_weights.get(name=Name.BIASES)
        biases_key = layer_name_plus_id + Name.BIASES.value

        self.result[weights_key] = weights
        self.result[biases_key] = biases

    def visit_batch_norm_test(self, layer: TestModeLayerWithWeightsAndParams):
        layer_name_plus_id = layer.get_name().value + str(layer.get_id())
        layer_weights = layer.get_weights()
        layer_params = layer.get_params()

        gamma = layer_weights.get(name=Name.GAMMA)
        gamma_key = layer_name_plus_id + Name.GAMMA.value

        beta = layer_weights.get(name=Name.BETA)
        beta_key = layer_name_plus_id + Name.BETA.value

        running_mean = layer_params.get(name=Name.RUNNING_MEAN)
        running_mean_key = layer_name_plus_id + Name.RUNNING_MEAN.value

        running_variance = layer_params.get(name=Name.RUNNING_VARIANCE)
        running_variance_key = layer_name_plus_id + Name.RUNNING_VARIANCE.value

        self.result[gamma_key] = gamma
        self.result[beta_key] = beta
        self.result[running_mean_key] = running_mean
        self.result[running_variance_key] = running_variance

    def get_result(self) -> OrderedDict:
        return self.result
