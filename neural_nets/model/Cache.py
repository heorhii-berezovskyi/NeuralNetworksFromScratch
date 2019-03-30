from numpy import ndarray

from neural_nets.model.Exception import ParamAlreadyExistsException, ParamNotFoundException
from neural_nets.model.Name import Name


class Cache:
    def __init__(self):
        self.params = {}

    def add(self, name: Name, value: ndarray):
        if name not in self.params:
            self.params[name] = value
        else:
            raise ParamAlreadyExistsException(
                'Param ' + name.value + ' already exists in params.')

    def get(self, name: Name):
        if name in self.params:
            return self.params[name]
        else:
            for p in self.params.keys():
                print(p)
            raise ParamNotFoundException(name.value + ' not found in params.')

    def update(self, name: Name, value: ndarray):
        if name in list(self.params):
            self.params[name] = value
        else:
            raise ParamNotFoundException(name.value + ' not found in params. Unable to update.')
