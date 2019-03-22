import numpy as np
from numpy import ndarray

from neural_nets.model.Loss import Loss
from neural_nets.model.Name import Name
from neural_nets.model.Params import Params


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def eval_data_loss(self, labels: ndarray, model_forward_run: list):
        scores = model_forward_run[-1].get(Name.OUTPUT)
        loss_run = Params()
        loss_run.add(name=Name.LABELS, value=labels)
        # Subtracting min values from scores for numeric stability.
        scores -= np.amax(scores, axis=0)

        exp_scores = np.exp(scores)
        # Calculating probabilities for each class over a mini-batch.
        probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
        loss_run.add(name=Name.PROBS, value=probs)

        # Losses of each image.
        correct_logprobs = -np.log(probs[labels, range(labels.size)])

        # Loss over a mini-batch.
        data_loss = np.sum(correct_logprobs) / labels.size
        return data_loss, loss_run

    def eval_gradient(self, loss_run: Params):
        labels = loss_run.get(name=Name.LABELS)
        dL_dLi = 1.0 / labels.size

        # dLi_scores = probs[k] - 1(yi = k)
        dLi_dscores = loss_run.get(name=Name.PROBS)
        dLi_dscores[labels, range(labels.size)] -= 1.0

        dL_dscores = dL_dLi * dLi_dscores
        return dL_dscores
