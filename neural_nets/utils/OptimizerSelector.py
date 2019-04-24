from neural_nets.model.Exception import ParamNotFoundException
from neural_nets.optimizer.Adagrad import Adagrad
from neural_nets.optimizer.Adam import Adam
from neural_nets.optimizer.RMSprop import RMSprop
from neural_nets.optimizer.SGDMomentum import SGDMomentum
from neural_nets.optimizer.SGDNesterovMomentum import SGDNesterovMomentum


class OptimizerSelector:
    def __init__(self):
        self.optimizer_dict = self._fill()

    def select(self, name: str):
        if name in self.optimizer_dict.keys():
            return self.optimizer_dict[name]
        else:
            raise ParamNotFoundException('Optimizer with name ' + name + ' not found.')

    @staticmethod
    def _fill() -> dict:
        return dict(adam=Adam,
                    adagrad=Adagrad,
                    rmsprop=RMSprop,
                    sgd_momentum=SGDMomentum,
                    sgd_nesterov_momentum=SGDNesterovMomentum)
