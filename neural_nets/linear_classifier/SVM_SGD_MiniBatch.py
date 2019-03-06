from numpy import argmax
from numpy import maximum
from numpy import mean
from numpy import ndarray
from numpy import sum
from numpy import zeros
from numpy import dot
from numpy.random import randn
from numpy import empty

from neural_nets.dataset.DatasetLoader import DatasetLoader

from neural_nets.utils.DatasetProcessingUtils import preprocess_dataset
from neural_nets.utils.DatasetProcessingUtils import sample
from neural_nets.utils.DatasetProcessingUtils import split_into_labels_and_data
from neural_nets.utils.PlotUtils import plot

import argparse


class SVM_SGD_MiniBatch_Classifier:
    """
    Linear model_draft with SVM loss. Trained with a Mini-batch Stochastic Gradient Descent optimization algorithm.
    """

    def __init__(self, batch_size: int, num_of_classes: int, size_of_input_vector: int, delta: float, reg: float):
        """
        :param batch_size: is a specified number of training samples per batch.
        :param delta: is a specified margin in the loss function.
        :param reg: is a specified L2 regularization strength.
        """
        self.W = 0.01 * randn(num_of_classes, size_of_input_vector)
        self.b = zeros((num_of_classes, 1))
        self.delta = delta
        self.reg = reg
        self.margins = empty((num_of_classes, batch_size))

    def eval_loss(self, labels: ndarray, images: ndarray):
        scores = dot(self.W, images.T) + self.b
        self.margins = maximum(0.0, scores[:, range(labels.size)] - scores[labels, range(labels.size)] + self.delta)
        self.margins[labels, range(labels.size)] = 0.0

        data_loss = self.margins.sum() / labels.size
        reg_loss = 0.5 * self.reg * sum(self.W * self.W)

        loss = data_loss + reg_loss
        return loss

    def eval_gradient(self, labels: ndarray, images: ndarray):
        indicators = self.margins
        indicators[indicators > 0.0] = 1.0
        indicators[labels, range(labels.size)] = -indicators[:, range(labels.size)].sum(axis=0)
        indicators /= labels.size

        dW = dot(indicators, images)
        db = sum(indicators, axis=1, keepdims=True)

        dW += self.reg * self.W
        return dW, db

    def train(self, labels: ndarray, images: ndarray, lr: float):
        dW, db = self.eval_gradient(labels, images)

        self.W -= lr * dW
        self.b -= lr * db

    def test(self, labels: ndarray, images: ndarray):
        global_scores = dot(self.W, images.T) + self.b
        predicted_labels = argmax(global_scores, axis=0)
        accuracy = mean(predicted_labels == labels)
        return accuracy


def run(args):
    loader = DatasetLoader(args.directory)
    train_dataset, test_dataset = loader.load(args.train_dataset_name, args.test_dataset_name)
    train_dataset = preprocess_dataset(train_dataset)

    test_dataset = preprocess_dataset(test_dataset)
    test_labels, test_data = split_into_labels_and_data(test_dataset)

    classifier = SVM_SGD_MiniBatch_Classifier(batch_size=args.batch_size, num_of_classes=args.num_of_classes,
                                              size_of_input_vector=args.size_of_input_vector,
                                              delta=args.delta, reg=args.reg)

    losses = []
    test_accuracies = []

    learning_rates = [0.0001, 0.0001 / 2.0, 0.0001 / 4.0, 0.0001 / 8.0, 0.0001 / 16.0]
    for lr in learning_rates:
        for i in range(10000):
            batch = sample(train_dataset, args.batch_size)
            label_batch, image_batch = split_into_labels_and_data(batch)
            losses.append(classifier.eval_loss(label_batch, image_batch))
            classifier.train(label_batch, image_batch, lr)
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
    parser.add_argument('--delta', type=float, help='Margin value in SVM loss function.')

    _args = parser.parse_args()

    run(_args)
