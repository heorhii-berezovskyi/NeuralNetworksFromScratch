import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TrainModeLayerWithWeights
from neural_nets.model.Model import TrainModel
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TrainLayerBaseVisitor
from neural_nets.optimizer.Optimizer import Optimizer


class AdamCacheInitVisitor(TrainLayerBaseVisitor):
    def __init__(self):
        self.cache_dict = {}

    def visit_affine_train(self, layer: TrainModeLayerWithWeights):
        layer_gradients_cache = Cache()
        weights = layer.get_weights()

        layer_gradients_cache.add(name=Name.ADAM_M_WEIGHTS,
                                  value=np.zeros_like(weights.get(name=Name.WEIGHTS), dtype=np.float64))
        layer_gradients_cache.add(name=Name.ADAM_V_WEIGHTS,
                                  value=np.zeros_like(weights.get(name=Name.WEIGHTS), dtype=np.float64))

        layer_gradients_cache.add(name=Name.ADAM_M_BIASES,
                                  value=np.zeros_like(weights.get(name=Name.BIASES), dtype=np.float64))
        layer_gradients_cache.add(name=Name.ADAM_V_BIASES,
                                  value=np.zeros_like(weights.get(name=Name.BIASES), dtype=np.float64))

        self.cache_dict[layer.get_id()] = layer_gradients_cache

    def visit_weightless_train(self, layer: TrainModeLayer):
        pass

    def visit_batch_norm_train(self, layer: TrainModeLayerWithWeights):
        layer_gradients_cache = Cache()
        weights = layer.get_weights()

        layer_gradients_cache.add(name=Name.ADAM_M_GAMMA,
                                  value=np.zeros_like(weights.get(name=Name.GAMMA), dtype=np.float64))
        layer_gradients_cache.add(name=Name.ADAM_V_GAMMA,
                                  value=np.zeros_like(weights.get(name=Name.GAMMA), dtype=np.float64))

        layer_gradients_cache.add(name=Name.ADAM_M_BETA,
                                  value=np.zeros_like(weights.get(name=Name.BETA), dtype=np.float64))
        layer_gradients_cache.add(name=Name.ADAM_V_BETA,
                                  value=np.zeros_like(weights.get(name=Name.BETA), dtype=np.float64))

        self.cache_dict[layer.get_id()] = layer_gradients_cache

    def get_grads_cache(self) -> dict:
        return self.cache_dict


class AdamWeightsUpdateVisitor(TrainLayerBaseVisitor):
    def __init__(self, learning_rate: np.float64, beta1: np.float64, beta2: np.float64, model_backward_run: list,
                 grads_cache: dict):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.model_backward_run = model_backward_run
        self.grads_cache = grads_cache

        self.num_iter = 1

    def visit_affine_train(self, layer: TrainModeLayerWithWeights):
        weight_names = [Name.WEIGHTS, Name.BIASES]
        grad_names = [Name.D_WEIGHTS, Name.D_BIASES]

        m_cache_names = [Name.ADAM_M_WEIGHTS, Name.ADAM_M_BIASES]
        v_cache_names = [Name.ADAM_V_WEIGHTS, Name.ADAM_V_BIASES]

        self._update(weight_names=weight_names,
                     grad_names=grad_names,
                     m_cache_names=m_cache_names,
                     v_cache_names=v_cache_names,
                     layer=layer)

    def visit_weightless_train(self, layer: TrainModeLayer):
        self.model_backward_run.pop()

    def visit_batch_norm_train(self, layer: TrainModeLayerWithWeights):
        weight_names = [Name.GAMMA, Name.BETA]
        grad_names = [Name.D_GAMMA, Name.D_BETA]

        m_cache_names = [Name.ADAM_M_GAMMA, Name.ADAM_M_BETA]
        v_cache_names = [Name.ADAM_V_GAMMA, Name.ADAM_V_BETA]

        self._update(weight_names=weight_names,
                     grad_names=grad_names,
                     m_cache_names=m_cache_names,
                     v_cache_names=v_cache_names,
                     layer=layer)

    def _update(self, weight_names: list, grad_names: list, m_cache_names: list, v_cache_names: list,
                layer: TrainModeLayerWithWeights):
        layer_id = layer.get_id()
        layer_backward_run = self.model_backward_run.pop()
        layer_weights = layer.get_weights()

        for wight_name, grad_name, m_name, v_name in zip(weight_names, grad_names, m_cache_names, v_cache_names):
            learning_param = layer_weights.get(name=wight_name)
            param_grad = layer_backward_run.get(grad_name)

            m = self.grads_cache[layer_id].get(m_name)
            m = self.beta1 * m + (1. - self.beta1) * param_grad
            self.grads_cache[layer_id].update(m_name, m)
            mt = m / (1. - self.beta1 ** self.num_iter)

            v = self.grads_cache[layer_id].get(v_name)
            v = self.beta2 * v + (1. - self.beta2) * (param_grad ** 2)
            self.grads_cache[layer_id].update(v_name, v)
            vt = v / (1. - self.beta2 ** self.num_iter)

            learning_param += -self.lr * mt / (np.sqrt(vt) + 1e-8)
            layer.get_weights().update(name=wight_name, value=learning_param)

            self.num_iter += 1


class Adam(Optimizer):
    def __init__(self, model: TrainModel, learning_rate: np.float64, beta1: np.float64, beta2: np.float64):
        super().__init__(model=model)
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.grads_cache = self.init_cache()

    def init_cache(self) -> dict:
        visitor = AdamCacheInitVisitor()
        for layer in self.model.get_layers():
            layer.accept(visitor)
        return visitor.get_grads_cache()

    def step(self, model_backward_run: list):
        model_backward_run.reverse()
        visitor = AdamWeightsUpdateVisitor(learning_rate=self.lr,
                                           beta1=self.beta1,
                                           beta2=self.beta2,
                                           model_backward_run=model_backward_run,
                                           grads_cache=self.grads_cache)
        for layer in reversed(self.model.get_layers()):
            layer.accept(visitor)
