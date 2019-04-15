from collections import OrderedDict

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TestModeLayer, TestModeLayerWithWeights, TestModeLayerWithWeightsAndParams
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TestLayerVisitor


class TestModelSaveVisitor(TestLayerVisitor):
    """
    Initializes a dict of parameters required to build a test model.
    """

    def __init__(self):
        self.result = OrderedDict()

    def visit_weightless_test(self, layer: TestModeLayer):
        pass

    def visit_affine_test(self, layer: TestModeLayerWithWeights):
        layer_id = layer.get_name().value + str(layer.get_id())
        self._save(data=layer.get_weights(), item_names=[Name.WEIGHTS, Name.BIASES], layer_name=layer_id)

    def visit_batch_norm_test(self, layer: TestModeLayerWithWeightsAndParams):
        layer_id = layer.get_name().value + str(layer.get_id())
        self._save(data=layer.get_weights(), item_names=[Name.GAMMA, Name.BETA], layer_name=layer_id)
        self._save(data=layer.get_params(), item_names=[Name.RUNNING_MEAN, Name.RUNNING_VAR], layer_name=layer_id)

    def _save(self, data: Cache, item_names: list, layer_name: str):
        for item_name in item_names:
            data_value = data.get(name=item_name)
            data_key = layer_name + item_name.value
            self.result[data_key] = data_value

    def get_result(self) -> OrderedDict:
        return self.result
