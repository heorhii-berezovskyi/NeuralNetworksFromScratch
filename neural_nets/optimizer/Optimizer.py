from abc import ABCMeta, abstractmethod

from neural_nets.model.Cache import Cache
from neural_nets.model.Loss import Loss
from neural_nets.model.Model import TrainModel

NOT_IMPLEMENTED = "You should implement this."


class Optimizer:
    """
    Optimizer representative. Optimizer updates train model weights.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model: TrainModel):
        self.model = model

    def backward_on_model(self, loss_function: Loss, model_forward_run: list, loss_run: Cache) -> list:
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

