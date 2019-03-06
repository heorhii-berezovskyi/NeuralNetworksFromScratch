from numpy import ndarray
from numpy import argmax
from numpy import maximum
from numpy import amax
from numpy import exp
from numpy import log
from numpy import mean
from numpy import zeros
from numpy import empty
from numpy import sum
from numpy import dot

from numpy.random import randn

from neural_nets.dataset.DatasetLoader import DatasetLoader

from neural_nets.utils.DatasetProcessingUtils import preprocess_dataset
from neural_nets.utils.DatasetProcessingUtils import split_into_labels_and_data
from neural_nets.utils.DatasetProcessingUtils import sample
from neural_nets.utils.PlotUtils import plot

import argparse


class TwoLayerNNCrossEntropy:
    """
    Two-layer neural network with ReLU activation function in the hidden layer, cross-entropy loss function.
    Network is trained with Mini-batch Stochastic Gradient Descent optimization algorithm.
    """

    def __init__(self, batch_size: int, num_of_classes: int, size_of_input_vector: int, num_of_hidden_neurons: int,
                 reg: float):
        """
        :param batch_size: is a specified number of training samples per batch.
        :param reg: is a specified L2 regularization strength.
        """
        self.W = 0.01 * randn(num_of_hidden_neurons, size_of_input_vector)
        self.b = zeros((num_of_hidden_neurons, 1))
        self.W2 = 0.01 * randn(num_of_classes, num_of_hidden_neurons)
        self.b2 = zeros((num_of_classes, 1))
        self.reg = reg
        self.hidden_layer = empty((num_of_hidden_neurons, batch_size))
        self.probs = empty((num_of_classes, batch_size))

    def eval_loss(self, label_batch: ndarray, image_batch: ndarray):
        self.hidden_layer = maximum(0.0, dot(self.W, image_batch.T) + self.b)
        scores = dot(self.W2, self.hidden_layer) + self.b2
        scores -= amax(scores, axis=0)

        exp_scores = exp(scores)
        self.probs = exp_scores / sum(exp_scores, axis=0, keepdims=True)

        correct_logprobs = -log(self.probs[label_batch, range(label_batch.size)])
        data_loss = sum(correct_logprobs) / label_batch.size
        reg_loss = 0.5 * self.reg * sum(self.W * self.W) + 0.5 * self.reg * sum(self.W2 * self.W2)
        loss = data_loss + reg_loss
        return loss

    def eval_gradient(self, label_batch: ndarray, image_batch: ndarray):
        dscores = self.probs
        dscores[label_batch, range(label_batch.size)] -= 1.0
        dscores /= label_batch.size

        dW2 = dot(dscores, self.hidden_layer.T)
        db2 = sum(dscores, axis=1, keepdims=True)

        dhidden = dot(self.W2.T, dscores)
        dhidden[self.hidden_layer <= 0.0] = 0.0

        dW = dot(dhidden, image_batch)
        db = sum(dhidden, axis=1, keepdims=True)

        dW2 += self.reg * self.W2
        dW += self.reg * self.W
        return dW, db, dW2, db2

    def train_iter(self, label_batch: ndarray, image_batch: ndarray, lr: float):
        dW, db, dW2, db2 = self.eval_gradient(label_batch, image_batch)

        self.W -= lr * dW
        self.b -= lr * db
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def test(self, label_batch: ndarray, image_batch: ndarray):
        hidden_layer = maximum(0.0, dot(self.W, image_batch.T) + self.b)
        scores = dot(self.W2, hidden_layer) + self.b2
        predicted_class = argmax(scores, axis=0)
        accuracy = mean(predicted_class == label_batch)
        return accuracy


def run(args):
    loader = DatasetLoader(args.directory)
    train_dataset, test_dataset = loader.load(args.train_dataset_name, args.test_dataset_name)

    train_dataset = preprocess_dataset(train_dataset)

    test_dataset = preprocess_dataset(test_dataset)
    test_labels, test_data = split_into_labels_and_data(test_dataset)

    classifier = TwoLayerNNCrossEntropy(batch_size=args.batch_size, num_of_classes=args.num_of_classes,
                                        size_of_input_vector=args.size_of_input_vector,
                                        num_of_hidden_neurons=args.num_of_hidden_neurons, reg=args.reg)

    learning_rates = [0.0001, 0.0001 / 2.0, 0.0001 / 4.0, 0.0001 / 8.0, 0.0001 / 16.0]

    losses = []
    test_accuracies = []

    for lr in learning_rates:
        for i in range(10000):
            batch = sample(train_dataset, args.batch_size)
            label_batch, image_batch = split_into_labels_and_data(batch)
            losses.append(classifier.eval_loss(label_batch, image_batch))
            classifier.train_iter(label_batch, image_batch, lr)
        test_accuracies.append(classifier.test(test_labels, test_data))

    print('Test Accuracy:', classifier.test(test_labels, test_data))

    plot(losses=losses, test_accuracies=test_accuracies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help='Directory leading to test and train datasets.')
    parser.add_argument('--train_dataset_name', type=str, help='Name of train dataset with extension.')
    parser.add_argument('--test_dataset_name', type=str, help='Name of test dataset with extension.')
    parser.add_argument('--batch_size', type=int, help='Number of training samples per batch.')
    parser.add_argument('--num_of_classes', type=int, help='Number of classes.')
    parser.add_argument('--size_of_input_vector', type=int, help='Length of an input array.')
    parser.add_argument('--num_of_hidden_neurons', type=int, help='Number of neurons in a hidden layer.')
    parser.add_argument('--reg', type=float, help='L2 regularization strength.')

    _args = parser.parse_args()

    run(_args)
