import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Loss import Loss
from neural_nets.model.Name import Name


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def eval_data_loss(self, labels: ndarray, model_forward_run: list) -> tuple:
        scores = model_forward_run[-1].get(Name.OUTPUT)
        shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = scores.shape[0]
        data_loss = -np.sum(log_probs[np.arange(N), labels]) / N

        loss_run = Cache()
        loss_run.add(name=Name.LABELS, value=labels)
        loss_run.add(name=Name.PROBS, value=probs)
        return data_loss, loss_run

    def eval_gradient(self, loss_run: Cache) -> ndarray:
        labels = loss_run.get(name=Name.LABELS)
        probs = loss_run.get(name=Name.PROBS)
        N = labels.size
        dinput = probs.copy()
        dinput[np.arange(N), labels] -= 1
        dinput /= N
        return dinput
