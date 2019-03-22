import numpy as np
from numpy import ndarray

from neural_nets.model.Loss import Loss
from neural_nets.model.Name import Name
from neural_nets.model.Params import Params


class SVMLoss(Loss):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def get_delta(self):
        return self.delta

    def eval_data_loss(self, labels: ndarray, model_forward_run: list):
        scores = model_forward_run[-1].get(Name.OUTPUT)
        loss_run = Params()
        loss_run.add(name=Name.LABELS, value=labels)
        margins = np.maximum(0.0, scores[:, range(labels.size)] - scores[labels, range(labels.size)] + self.delta)
        margins[labels, range(labels.size)] = 0.0
        loss_run.add(name=Name.MARGINS, value=margins)

        data_loss = margins.sum() / labels.size
        return data_loss, loss_run

    def eval_gradient(self, loss_run: Params):
        labels = loss_run.get(name=Name.LABELS)

        indicators = loss_run.get(name=Name.MARGINS)
        indicators[indicators > 0.0] = 1.0
        indicators[labels, range(labels.size)] = -indicators[:, range(labels.size)].sum(axis=0)

        dL_dLi = 1.0 / labels.size
        dLi_dscores = indicators

        dL_dscores = dL_dLi * dLi_dscores
        return dL_dscores
