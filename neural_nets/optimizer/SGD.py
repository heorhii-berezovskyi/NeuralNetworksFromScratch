from neural_nets.model.BatchNorm1D import BatchNorm1DTrain
from neural_nets.model.BatchNorm2D import BatchNorm2DTrain
from neural_nets.model.Conv2D import Conv2DTrain
from neural_nets.model.Dropout1D import Dropout1DTrain
from neural_nets.model.Dropout2D import Dropout2DTrain
from neural_nets.model.Layer import TrainModeLayerWithWeights
from neural_nets.model.Linear import LinearTrain
from neural_nets.model.MaxPool import MaxPoolTrain
from neural_nets.model.Model import TrainModel
from neural_nets.model.Name import Name
from neural_nets.model.Relu import ReluTrain
from neural_nets.model.Visitor import Visitor
from neural_nets.optimizer.Optimizer import Optimizer


class SGDWeightsUpdateVisitor(Visitor):
    """
    Updates weights on each type of train mode model layers through a stochastic gradient update procedure.
    """

    def __init__(self, learning_rate: float, model_backward_run: list):
        self.lr = learning_rate
        self.model_backward_run = model_backward_run

    def visit_linear(self, layer: LinearTrain):
        weight_names = [Name.WEIGHTS, Name.BIASES]
        grad_names = [Name.D_WEIGHTS, Name.D_BIASES]
        self._update(weight_names=weight_names, grad_names=grad_names, layer=layer)

    def visit_conv2d(self, layer: Conv2DTrain):
        weight_names = [Name.KERNEL_WEIGHTS, Name.KERNEL_BIASES]
        grad_names = [Name.D_KERNEL_WEIGHTS, Name.D_KERNEL_BIASES]
        self._update(weight_names=weight_names, grad_names=grad_names, layer=layer)

    def visit_relu(self, layer: ReluTrain):
        self.model_backward_run.pop()

    def visit_max_pool(self, layer: MaxPoolTrain):
        self.model_backward_run.pop()

    def visit_dropout1d(self, layer: Dropout1DTrain):
        self.model_backward_run.pop()

    def visit_dropout2d(self, layer: Dropout2DTrain):
        self.model_backward_run.pop()

    def visit_batch_norm_1d(self, layer: BatchNorm1DTrain):
        weight_names = [Name.GAMMA, Name.BETA]
        grad_names = [Name.D_GAMMA, Name.D_BETA]
        self._update(weight_names=weight_names, grad_names=grad_names, layer=layer)

    def visit_batch_norm_2d(self, layer: BatchNorm2DTrain):
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
    def __init__(self, model: TrainModel, learning_rate: float):
        super().__init__(model=model)
        self.lr = learning_rate

    def step(self, model_backward_run: list):
        model_backward_run.reverse()
        visitor = SGDWeightsUpdateVisitor(learning_rate=self.lr, model_backward_run=model_backward_run)
        for layer in reversed(self.model.get_layers()):
            layer.accept(visitor)
