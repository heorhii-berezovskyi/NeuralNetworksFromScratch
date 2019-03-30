import numpy as np
from numpy import ndarray

from neural_nets.model.Loss import Loss
from neural_nets.model.Name import Name
from neural_nets.model.Cache import Cache


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def eval_data_loss(self, labels: ndarray, model_forward_run: list):
        input_data = model_forward_run[-1].get(Name.OUTPUT)
        probs = input_data - np.max(input_data, axis=1, keepdims=True)
        probs = np.exp(probs)
        probs[probs == 0.0] += 10 ** -7
        probs /= np.sum(probs, axis=1, keepdims=True)
        num_of_samples = labels.size
        data_loss = -np.sum(np.log(probs[np.arange(num_of_samples), labels])) / float(num_of_samples)

        loss_run = Cache()
        loss_run.add(name=Name.LABELS, value=labels)
        loss_run.add(name=Name.PROBS, value=probs)
        return data_loss, loss_run

    def eval_gradient(self, loss_run: Cache):
        labels = loss_run.get(name=Name.LABELS)
        probs = loss_run.get(name=Name.PROBS)
        dinput = probs.copy()
        num_of_samples = labels.size
        dinput[np.arange(num_of_samples), labels] -= 1.0
        dinput /= float(num_of_samples)
        return dinput
