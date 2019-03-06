from numpy import amax
from numpy import argmax
from numpy import dot
from numpy import exp
from numpy import log
from numpy import mean
from numpy import ndarray
from numpy import sum
from numpy import zeros
from numpy import empty
from numpy.random import rand

from neural_nets.dataset.DatasetLoader import DatasetLoader

from neural_nets.utils.DatasetProcessingUtils import preprocess_dataset
from neural_nets.utils.DatasetProcessingUtils import sample
from neural_nets.utils.DatasetProcessingUtils import split_into_labels_and_data
from neural_nets.utils.PlotUtils import plot

import argparse


class Softmax_SGD_MiniBatch_Classifier:
    """
    Linear model_draft with cross-entropy loss. Trained with a Mini-batch Stochastic Gradient Descent optimization algorithm.
    """

    def __init__(self, batch_size: int, num_of_classes: int, size_of_input_vector: int, reg: float):
        """
        :param batch_size: is a specified number of training samples per batch.
        :param reg: is a specified L2 regularization strength.
        """
        self.W = rand(num_of_classes, size_of_input_vector) * 0.01
        self.b = zeros((num_of_classes, 1))
        self.reg = reg
        self.probs = empty((num_of_classes, batch_size))

    def eval_loss(self, label_batch: ndarray, image_batch: ndarray):
        scores = dot(self.W, image_batch.T) + self.b
        scores -= amax(scores, axis=0)
        exp_scores = exp(scores)
        self.probs = exp_scores / sum(exp_scores, axis=0, keepdims=True)
        correct_logprobs = -log(self.probs[label_batch, range(label_batch.size)])

        data_loss = sum(correct_logprobs) / label_batch.size
        reg_loss = 0.5 * self.reg * sum(self.W * self.W)

        loss = data_loss + reg_loss
        return loss

    def eval_gradient(self, label_batch: ndarray, image_batch: ndarray):
        dscores = self.probs
        dscores[label_batch, range(label_batch.size)] -= 1.0
        dscores /= label_batch.size

        dW = dot(dscores, image_batch)
        db = sum(dscores, axis=1, keepdims=True)

        dW += self.reg * self.W
        return dW, db

    def train_epoch(self, label_batch: ndarray, image_batch: ndarray, lr: float):
        dW, db = self.eval_gradient(label_batch, image_batch)

        self.W -= lr * dW
        self.b -= lr * db

    def test(self, test_labels: ndarray, test_data: ndarray):
        global_scores = dot(self.W, test_data.T) + self.b
        predicted_labels = argmax(global_scores, axis=0)
        accuracy = mean(predicted_labels == test_labels)
        return accuracy


def run(args):
    loader = DatasetLoader(args.directory)
    train_dataset, test_dataset = loader.load(train_dataset_name=args.train_dataset_name,
                                              test_dataset_name=args.test_dataset_name)

    train_dataset = preprocess_dataset(train_dataset)

    test_dataset = preprocess_dataset(test_dataset)
    test_labels, test_data = split_into_labels_and_data(test_dataset)

    classifier = Softmax_SGD_MiniBatch_Classifier(batch_size=args.batch_size, num_of_classes=args.num_of_classes,
                                                  size_of_input_vector=args.size_of_input_vector, reg=args.reg)

    losses = []
    test_accuracies = []

    learning_rates = [0.0001, 0.0001 / 2.0, 0.0001 / 4.0, 0.0001 / 8.0, 0.0001 / 16.0]
    for lr in learning_rates:
        for i in range(10000):
            batch = sample(train_dataset, args.batch_size)
            label_batch, image_batch = split_into_labels_and_data(batch)
            losses.append(classifier.eval_loss(label_batch, image_batch))
            classifier.train_epoch(label_batch, image_batch, lr)
        test_accuracies.append(classifier.test(test_labels, test_data))

    print('Accuracy:', classifier.test(test_labels, test_data))

    plot(losses=losses, test_accuracies=test_accuracies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help='Directory leading to test and train datasets.')
    parser.add_argument('--train_dataset_name', type=str, help='Name of train dataset with extension.')
    parser.add_argument('--test_dataset_name', type=str, help='Name of test dataset with extension.')
    parser.add_argument('--batch_size', type=int, help='Number of training samples per batch.')
    parser.add_argument('--num_of_classes', type=int, help='Number of classes.')
    parser.add_argument('--size_of_input_vector', type=int, help='Length of an input array.')
    parser.add_argument('--reg', type=float, help='L2 regularization strength.')

    _args = parser.parse_args()

    run(_args)
