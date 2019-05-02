import importlib

from neural_nets.model.Exception import ParamNotFoundException
from neural_nets.model.Model import TrainModel


class ModelSelector:
    @staticmethod
    def select(name: str) -> TrainModel:
        try:
            name = '.' + name
            module = importlib.import_module(name=name, package='neural_nets.run.models')
            return module.net()
        except:
            raise ParamNotFoundException('Model with name ' + name + ' not found.')

