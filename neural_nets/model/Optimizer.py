from abc import ABCMeta, abstractmethod

from neural_nets.model.Loss import Loss
from neural_nets.model.Model import TrainModel
from neural_nets.model.Params import Params
from neural_nets.model.Visitor import SGDWeightsUpdateVisitor, SGDMomentumParamsInitVisitor, \
    SGDMomentumWeightsUpdateVisitor, \
    SGDNesterovMomentumParamsInitVisitor, SGDNesterovMomentumWeightsUpdateVisitor

NOT_IMPLEMENTED = "You should implement this."


class Optimizer:
    """
    Optimizer representative. Optimizer updates train model weights.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model: TrainModel):
        self.model = model

    def backward(self, loss_function: Loss, model_forward_run: list, loss_run: Params):
        """
        Performs backward pass of a train model based on the loss function, model forward run and loss function run.
        :param loss_function: is a loss function.
        :param model_forward_run: is a list of model forward run parameters.
        :param loss_run: is an object storing loss function run parameters.
        :return: model backward run.
        """
        dout = loss_function.eval_gradient(loss_run=loss_run)
        model_backward_run = []
        for layer, layer_forward_run in zip(reversed(self.model.get_layers()), reversed(model_forward_run)):
            dout, layer_backward_run = layer.backward(dout, layer_forward_run)
            model_backward_run.append(layer_backward_run)
        return model_backward_run

    @abstractmethod
    def step(self, model_backward_run: list):
        """
        Performs train model weights update based on the model backward run.
        :param model_backward_run: is a list of a train model backward run parameters.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)


class SGD(Optimizer):
    def __init__(self, model: TrainModel, learning_rate: float):
        super().__init__(model=model)
        self.lr = learning_rate

    def step(self, model_backward_run: list):
        model_backward_run.reverse()
        visitor = SGDWeightsUpdateVisitor(learning_rate=self.lr, model_backward_run=model_backward_run)
        for layer in reversed(self.model.get_layers()):
            layer.accept(visitor)


class SGDMomentum(Optimizer):
    def __init__(self, model: TrainModel, learning_rate: float, mu: float):
        super().__init__(model=model)
        self.lr = learning_rate
        self.mu = mu
        self.velocity_params = self.init_params()

    def init_params(self):
        visitor = SGDMomentumParamsInitVisitor()
        for layer in self.model.get_layers():
            layer.accept(visitor)
        return visitor.get_velocity_params()

    def step(self, model_backward_run: list):
        model_backward_run.reverse()
        visitor = SGDMomentumWeightsUpdateVisitor(learning_rate=self.lr, mu=self.mu,
                                                  model_backward_run=model_backward_run,
                                                  velocity_params=self.velocity_params)
        for layer in reversed(self.model.get_layers()):
            layer.accept(visitor)


class SGDNesterovMomentum(Optimizer):
    def __init__(self, model: TrainModel, learning_rate: float, mu: float):
        super().__init__(model=model)
        self.lr = learning_rate
        self.mu = mu
        self.velocity_params = self.init_params()

    def init_params(self):
        visitor = SGDNesterovMomentumParamsInitVisitor()
        for layer in self.model.get_layers():
            layer.accept(visitor)
        return visitor.get_velocity_params()

    def step(self, model_backward_run: list):
        model_backward_run.reverse()
        visitor = SGDNesterovMomentumWeightsUpdateVisitor(learning_rate=self.lr, mu=self.mu,
                                                          model_backward_run=model_backward_run,
                                                          velocity_params=self.velocity_params)
        for layer in reversed(self.model.get_layers()):
            layer.accept(visitor)
