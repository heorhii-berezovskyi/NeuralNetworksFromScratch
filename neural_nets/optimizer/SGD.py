from neural_nets.model.Layer import TrainModeLayer, TrainModeLayerWithWeights
from neural_nets.model.Model import TrainModel
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TrainLayerBaseVisitor
from neural_nets.optimizer.Optimizer import Optimizer
import numpy as np


class SGDWeightsUpdateVisitor(TrainLayerBaseVisitor):
    """
    Updates weights on each type of train mode model layers through a stochastic gradient update procedure.
    """

    def __init__(self, learning_rate: np.float64, model_backward_run: list):
        self.lr = learning_rate
        self.model_backward_run = model_backward_run

    def visit_affine_train(self, layer: TrainModeLayerWithWeights):
        weight_names = [Name.WEIGHTS, Name.BIASES]
        grad_names = [Name.D_WEIGHTS, Name.D_BIASES]
        self._update(weight_names=weight_names, grad_names=grad_names, layer=layer)

    def visit_weightless_train(self, layer: TrainModeLayer):
        self.model_backward_run.pop()

    def visit_batch_norm_train(self, layer: TrainModeLayerWithWeights):
        weight_names = [Name.GAMMA, Name.BETA]
        grad_names = [Name.D_GAMMA, Name.D_BETA]
        self._update(weight_names=weight_names, grad_names=grad_names, layer=layer)

    def _update(self, weight_names: list, grad_names: list, layer: TrainModeLayerWithWeights):
        layer_backward_run = self.model_backward_run.pop()
        weights = layer.get_weights()
        for weight_name, grad_name in zip(weight_names, grad_names):
            learning_param = weights.get(name=weight_name)
            param_grad = layer_backward_run.get(grad_name)
            learning_param -= self.lr * param_grad
            layer.get_weights().update(name=weight_name, value=learning_param)


class SGD(Optimizer):
    def __init__(self, model: TrainModel, learning_rate: np.float64):
        super().__init__(model=model)
        self.lr = learning_rate

    def step(self, model_backward_run: list):
        model_backward_run.reverse()
        visitor = SGDWeightsUpdateVisitor(learning_rate=self.lr, model_backward_run=model_backward_run)
        for layer in reversed(self.model.get_layers()):
            layer.accept(visitor)
