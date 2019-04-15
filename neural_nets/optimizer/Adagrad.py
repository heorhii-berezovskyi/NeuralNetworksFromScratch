import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TrainModeLayerWithWeights
from neural_nets.model.Model import TrainModel
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TrainLayerVisitor
from neural_nets.optimizer.Optimizer import Optimizer


class AdagradMemoryInitVisitor(TrainLayerVisitor):
    def __init__(self):
        self.memory = []

    def visit_affine_train(self, layer: TrainModeLayerWithWeights):
        layer_gradients_cache = Cache()
        weights = layer.get_weights()

        layer_gradients_cache.add(name=Name.D_WEIGHTS_CACHE,
                                  value=np.zeros_like(weights.get(name=Name.WEIGHTS), dtype=np.float64))
        layer_gradients_cache.add(name=Name.D_BIASES_CACHE,
                                  value=np.zeros_like(weights.get(name=Name.BIASES), dtype=np.float64))
        self.memory.append(layer_gradients_cache)

    def visit_weightless_train(self, layer: TrainModeLayer):
        self.memory.append(Cache())

    def visit_batch_norm_train(self, layer: TrainModeLayerWithWeights):
        layer_gradients_cache = Cache()
        weights = layer.get_weights()

        layer_gradients_cache.add(name=Name.D_GAMMA_CACHE,
                                  value=np.zeros_like(weights.get(name=Name.GAMMA), dtype=np.float64))
        layer_gradients_cache.add(name=Name.D_BETA_CACHE,
                                  value=np.zeros_like(weights.get(name=Name.BETA), dtype=np.float64))
        self.memory.append(layer_gradients_cache)

    def get_result(self) -> list:
        return self.memory


class AdagradWeightsUpdateVisitor(TrainLayerVisitor):
    def __init__(self, learning_rate: float, model_backward_run: list, grads_cache: list):
        self.lr = learning_rate
        self.model_backward_run = model_backward_run
        self.grads_cache = grads_cache

    def visit_affine_train(self, layer: TrainModeLayerWithWeights):
        weight_names = [Name.WEIGHTS, Name.BIASES]
        grad_names = [Name.D_WEIGHTS, Name.D_BIASES]
        cache_names = [Name.D_WEIGHTS_CACHE, Name.D_BIASES_CACHE]
        self._update(weight_names=weight_names, grad_names=grad_names, cache_names=cache_names, layer=layer)

    def visit_weightless_train(self, layer: TrainModeLayer):
        self.model_backward_run.pop()

    def visit_batch_norm_train(self, layer: TrainModeLayerWithWeights):
        weight_names = [Name.GAMMA, Name.BETA]
        grad_names = [Name.D_GAMMA, Name.D_BETA]
        cache_names = [Name.D_GAMMA_CACHE, Name.D_BETA_CACHE]
        self._update(weight_names=weight_names, grad_names=grad_names, cache_names=cache_names, layer=layer)

    def _update(self, weight_names: list, grad_names: list, cache_names: list, layer: TrainModeLayerWithWeights):
        layer_id = layer.get_id()
        layer_backward_run = self.model_backward_run.pop()
        layer_weights = layer.get_weights()
        for wight_name, grad_name, cache_name in zip(weight_names, grad_names, cache_names):
            learning_param = layer_weights.get(name=wight_name)
            param_grad = layer_backward_run.get(grad_name)
            param_grad_cache = self.grads_cache[layer_id].get(cache_name)
            param_grad_cache += param_grad ** 2
            learning_param += -self.lr * param_grad / (np.sqrt(param_grad_cache) + 1e-6)

            self.grads_cache[layer_id].update(cache_name, param_grad_cache)
            layer.get_weights().update(name=wight_name, value=learning_param)


class Adagrad(Optimizer):
    def __init__(self, model: TrainModel, learning_rate: float):
        super().__init__(model=model)
        self.lr = learning_rate
        self.grads_cache = self.init_memory()

    def init_memory(self) -> list:
        visitor = AdagradMemoryInitVisitor()
        for layer in self.model.get_layers():
            layer.accept(visitor)
        return visitor.get_result()

    def step(self, model_backward_run: list):
        model_backward_run.reverse()
        visitor = AdagradWeightsUpdateVisitor(learning_rate=self.lr,
                                              model_backward_run=model_backward_run,
                                              grads_cache=self.grads_cache)
        for layer in reversed(self.model.get_layers()):
            layer.accept(visitor)
