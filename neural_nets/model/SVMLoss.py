import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Loss import Loss
from neural_nets.model.Name import Name


class SVMLoss(Loss):
    def __init__(self, delta: np.float64):
        self.delta = delta

    def eval_data_loss(self, labels: ndarray, scores: ndarray) -> tuple:
        num_of_samples = labels.size
        correct_class_scores = scores[np.arange(num_of_samples), labels]
        margins = np.maximum(0., scores - correct_class_scores[:, np.newaxis] + self.delta)
        margins[np.arange(num_of_samples), labels] = 0.
        data_loss = np.sum(margins) / num_of_samples

        loss_run = Cache()
        loss_run.add(name=Name.LABELS, value=labels)
        loss_run.add(name=Name.MARGINS, value=margins)
        return data_loss, loss_run

    def eval_gradient(self, loss_run: Cache) -> ndarray:
        labels = loss_run.get(name=Name.LABELS)
        margins = loss_run.get(name=Name.MARGINS)
        num_of_samples = labels.size
        num_pos = np.sum(margins > 0., axis=1)
        dinput = np.zeros_like(margins, dtype=float)
        dinput[margins > 0.] = 1.
        dinput[np.arange(num_of_samples), labels] -= num_pos
        dinput /= num_of_samples
        return dinput
