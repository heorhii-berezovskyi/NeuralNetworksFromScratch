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

    def get(self, name: Name) -> ndarray:
        if name in self.params:
            return self.params[name]
        else:
            raise ParamNotFoundException(name.value + ' not found in params.')

    def pop(self, name: Name) -> ndarray:
        if name in self.params:
            return self.params.pop(name)
        else:
            raise ParamNotFoundException(name.value + ' not found in params.')

    def get_keys(self):
        return self.params.keys()
