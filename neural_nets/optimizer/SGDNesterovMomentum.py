import numpy as np

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TrainModeLayerWithWeights
from neural_nets.model.Model import TrainModel
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import Visitor
from neural_nets.optimizer.Optimizer import Optimizer


class SGDNesterovMomentumParamsInitVisitor(Visitor):
    def __init__(self):
        self.params = {}

    def visit_linear(self, layer: TrainModeLayerWithWeights):
        layer_velocity = Cache()
        weights = layer.get_weights()

        layer_velocity.add(name=Name.V_WEIGHTS, value=np.zeros_like(weights.get(name=Name.WEIGHTS), dtype=np.float64))
        layer_velocity.add(name=Name.V_WEIGHTS_PREV,
                           value=np.zeros_like(weights.get(name=Name.WEIGHTS), dtype=np.float64))

        layer_velocity.add(name=Name.V_BIASES, value=np.zeros_like(weights.get(name=Name.BIASES), dtype=np.float64))
        layer_velocity.add(name=Name.V_BIASES_PREV,
                           value=np.zeros_like(weights.get(name=Name.BIASES), dtype=np.float64))

        self.params[layer.get_id()] = layer_velocity

    def visit_weightless_layer(self, layer: TrainModeLayer):
        pass

    def visit_batch_norm(self, layer: TrainModeLayerWithWeights):
        layer_velocity = Cache()
        weights = layer.get_weights()

        layer_velocity.add(name=Name.V_GAMMA, value=np.zeros_like(weights.get(name=Name.GAMMA), dtype=np.float64))
        layer_velocity.add(name=Name.V_GAMMA_PREV, value=np.zeros_like(weights.get(name=Name.GAMMA), dtype=np.float64))

        layer_velocity.add(name=Name.V_BETA, value=np.zeros_like(weights.get(name=Name.BETA), dtype=np.float64))
        layer_velocity.add(name=Name.V_BETA_PREV, value=np.zeros_like(weights.get(name=Name.BETA), dtype=np.float64))

        self.params[layer.get_id()] = layer_velocity

    def get_velocity_params(self) -> dict:
        return self.params


class SGDNesterovMomentumWeightsUpdateVisitor(Visitor):
    def __init__(self, learning_rate: np.float64, mu: np.float64, model_backward_run: list, velocity_params: dict):
        self.lr = learning_rate
        self.mu = mu
        self.model_backward_run = model_backward_run
        self.velocity_params = velocity_params

    def visit_linear(self, layer: TrainModeLayerWithWeights):
        weight_names = [Name.WEIGHTS, Name.BIASES]
        grad_names = [Name.D_WEIGHTS, Name.D_BIASES]
        v_names = [Name.V_WEIGHTS, Name.V_BIASES]
        v_prev_names = [Name.V_WEIGHTS_PREV, Name.V_BIASES_PREV]
        self._update(weight_names=weight_names, grad_names=grad_names, v_names=v_names, v_prev_names=v_prev_names,
                     layer=layer)

    def visit_weightless_layer(self, layer: TrainModeLayer):
        self.model_backward_run.pop()

    def visit_batch_norm(self, layer: TrainModeLayerWithWeights):
        weight_names = [Name.GAMMA, Name.BETA]
        grad_names = [Name.D_GAMMA, Name.D_BETA]
        v_names = [Name.V_GAMMA, Name.V_BETA]
        v_prev_names = [Name.V_GAMMA_PREV, Name.V_BETA_PREV]
        self._update(weight_names=weight_names, grad_names=grad_names, v_names=v_names, v_prev_names=v_prev_names,
                     layer=layer)

    def _update(self, weight_names: list, grad_names: list, v_names: list, v_prev_names: list,
                layer: TrainModeLayerWithWeights):
        layer_id = layer.get_id()
        layer_backward_run = self.model_backward_run.pop()
        layer_weights = layer.get_weights()
        for weight_name, grad_name, v_name, v_prev_name in zip(weight_names, grad_names, v_names, v_prev_names):
            learning_param = layer_weights.get(name=weight_name)
            param_grad = layer_backward_run.get(grad_name)
            param_v = self.velocity_params[layer_id].get(v_name)
            self.velocity_params[layer_id].update(v_prev_name, param_v)
            param_v = self.mu * param_v - self.lr * param_grad
            self.velocity_params[layer_id].update(v_name, param_v)
            learning_param += -self.mu * self.velocity_params[layer_id].get(v_prev_name) + (1.0 + self.mu) * param_v
            layer.get_weights().update(name=weight_name, value=learning_param)


class SGDNesterovMomentum(Optimizer):
    def __init__(self, model: TrainModel, learning_rate: np.float64, mu: np.float64):
        super().__init__(model=model)
        self.lr = learning_rate
        self.mu = mu
        self.velocity_params = self.init_params()

    def init_params(self) -> dict:
        visitor = SGDNesterovMomentumParamsInitVisitor()
        for layer in self.model.get_layers():
            layer.accept(visitor)
        return visitor.get_velocity_params()

    def step(self, model_backward_run: list):
        model_backward_run.reverse()
        visitor = SGDNesterovMomentumWeightsUpdateVisitor(learning_rate=self.lr,
                                                          mu=self.mu,
                                                          model_backward_run=model_backward_run,
                                                          velocity_params=self.velocity_params)
        for layer in reversed(self.model.get_layers()):
            layer.accept(visitor)