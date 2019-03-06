from abc import ABCMeta, abstractmethod

NOT_IMPLEMENTED = "You should implement this."


class Layer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, input_data: tuple):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def backward(self):
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def accept(self, visitor):
        raise NotImplementedError(NOT_IMPLEMENTED)
