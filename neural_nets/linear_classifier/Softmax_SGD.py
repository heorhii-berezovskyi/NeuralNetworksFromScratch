from numpy import argmax
from numpy import exp
from numpy import log
from numpy import mean
from numpy import ndarray
from numpy import outer
from numpy.random import randn
from numpy.random import shuffle
from numpy import sum
from numpy import empty
from numpy import zeros
from numpy import dot

from neural_nets.dataset.DatasetLoader import DatasetLoader

from neural_nets.utils.DatasetProcessingUtils import preprocess_dataset
from neural_nets.utils.DatasetProcessingUtils import split_into_labels_and_data
from neural_nets.utils.PlotUtils import plot

import argparse


class Softmax_SGD_Classifier:
    """
    Linear model_draft with cross-entropy loss. Trained with a classical Stochastic Gradient Descent optimizations algorithm.
    """

    def __init__(self, num_of_classes: int, size_of_input_vector: int, reg: float):
        """
        :param reg: is a specified L2 regularization strength.
        """
        self.W = randn(num_of_classes, size_of_input_vector) * 0.01
        self.b = zeros((num_of_classes, 1))
        self.reg = reg
        self.exp_scores = empty((num_of_classes, 1))

    def eval_image_loss(self, label: int, image: ndarray):
        image = image.reshape(len(image), 1)
        scores = dot(self.W, image) + self.b
        scores -= scores.max()
        self.exp_scores = exp(scores)

        prob = self.exp_scores[label] / self.exp_scores.sum()

        data_loss = -log(prob)
        reg_loss = 0.5 * self.reg * sum(self.W * self.W)

        loss = data_loss + reg_loss
        return loss

    def eval_image_gradient(self, label: int, image: ndarray):
        probs = self.exp_scores / self.exp_scores.sum()

        dscores = probs
        dscores[label] -= 1.0

        dW = outer(dscores, image)
        dW += self.reg * self.W

        db = dscores
        return dW, db

    def train_iteration(self, label: int, image: ndarray, lr: float):
        dW, db = self.eval_image_gradient(label, image)
        self.W -= lr * dW
        self.b -= lr * db

    def test(self, test_labels: ndarray, test_data: ndarray):
        global_scores = self.W.dot(test_data.T) + self.b
        predicted_labels = argmax(global_scores, axis=0)
        accuracy = mean(predicted_labels == test_labels)
        return accuracy


def run(args):
    loader = DatasetLoader(args.directory)
    train_dataset, test_dataset = loader.load(args.train_dataset_name, args.test_dataset_name)

    train_dataset = preprocess_dataset(train_dataset)

    test_dataset = preprocess_dataset(test_dataset)
    test_labels, test_data = split_into_labels_and_data(test_dataset)

    classifier = Softmax_SGD_Classifier(num_of_classes=args.num_of_classes,
                                        size_of_input_vector=args.size_of_input_vector, reg=args.reg)

    losses = []
    test_accuracies = []

    learning_rates = [0.0001 / 16.0, 0.0001 / 32.0, 0.0001 / 64.0, 0.0001 / 128.0, 0.0001 / 256.0]
    for lr in learning_rates:
        shuffle(train_dataset)
        train_labels, train_data = split_into_labels_and_data(train_dataset)
        for label, image in zip(train_labels, train_data):
            losses.append(classifier.eval_image_loss(label, image))
            classifier.train_iteration(label, image, lr)

        test_accuracies.append(classifier.test(test_labels, test_data))

    print('Accuracy:', classifier.test(test_labels, test_data))

    plot(losses=losses, test_accuracies=test_accuracies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help='Directory leading to test and train datasets.')
    parser.add_argument('--train_dataset_name', type=str, help='Name of train dataset with extension.')
    parser.add_argument('--test_dataset_name', type=str, help='Name of test dataset with extension.')
    parser.add_argument('--num_of_classes', type=int, help='Number of classes.')
    parser.add_argument('--size_of_input_vector', type=int, help='Length of an input array.')
    parser.add_argument('--reg', type=float, help='L2 regularization strength.')

    _args = parser.parse_args()

    run(_args)
