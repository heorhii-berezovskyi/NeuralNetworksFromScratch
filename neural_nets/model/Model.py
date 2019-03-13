import argparse

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from neural_nets.dataset.DatasetLoader import DatasetLoader
from neural_nets.model.BatchNorm import BatchNorm
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.model.Layer import Layer
from neural_nets.model.Linear import Linear
from neural_nets.model.Relu import Relu
from neural_nets.model.Loss import Loss
from neural_nets.model.Optimizer import SGD, SGDMomentum
from neural_nets.model.Visitor import RegularizationVisitor, ModeTuningVisitor
from neural_nets.utils.DatasetProcessingUtils import preprocess_dataset, sample, split_into_labels_and_data
from neural_nets.utils.PlotUtils import plot


class Model:
    """
    Neural net model representative class. Model can be comprised of different type, size and count of layers.
    """

    def __init__(self, reg: float, update_type: str, loss_function: Loss):
        self.layers = []
        self.reg = reg
        self.update_type = update_type
        self.loss_function = loss_function

        self.reg_visitor = RegularizationVisitor(reg_strength=reg)
        self.mode_visitor = ModeTuningVisitor()

    def add_layer(self, layer: Layer):
        """
        Adds layer to layers list.
        :param layer: is a model layer representative.
        """
        self.layers.append(layer)

    def forward(self, images: ndarray):
        """
        Computes forward propagation on each layer.
        :param images: is a numpy array of training images.
        :return: model scores.
        """
        input_data = images.T
        for layer in self.layers:
            output_data = layer.forward(input_data)
            input_data = output_data
        scores = input_data
        return scores

    def eval_loss(self, labels: ndarray, scores: ndarray):
        self.reg_visitor.reset()
        for layer in self.layers:
            layer.accept(self.reg_visitor)
        reg_loss = self.reg_visitor.get_reg_loss()
        data_loss = self.loss_function.eval_data_loss(labels=labels, scores=scores)
        loss = data_loss + reg_loss
        return loss

    def backward(self):
        """
        Performs backward propagation through layers.
        """

        dout = self.loss_function.eval_gradient()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def test(self, labels: ndarray, images: ndarray):
        """
        Evaluates accuracy of a trained model on test labels and data.
        :param labels: is a batch of labels.
        :param images: is a batch of images.
        """
        scores = self.forward(images=images)
        predicted_class = np.argmax(scores, axis=0)
        accuracy = np.mean(predicted_class == labels)
        return accuracy

    def set_mode(self, mode: str):
        self.mode_visitor.set_mode(mode)
        for layer in self.layers:
            layer.accept(self.mode_visitor)


def run(args):
    linear_layer1 = Linear(args.num_of_hidden_neurons, args.size_of_input_vector)
    batch_norm1 = BatchNorm(args.num_of_hidden_neurons, 0.9)
    relu_laye1 = Relu()
    linear_layer2 = Linear(args.num_of_classes, args.num_of_hidden_neurons)

    loss = CrossEntropyLoss()
    # loss = SVM_Loss(10.0)

    model = Model(reg=args.reg, update_type='momentum', loss_function=loss)
    model.add_layer(linear_layer1)
    model.add_layer(batch_norm1)
    model.add_layer(relu_laye1)
    model.add_layer(linear_layer2)

    # optimizer = SGD(model.layers, learning_rate=0.01, reg=args.reg)
    optimizer = SGDMomentum(model.layers, learning_rate=0.01, reg=args.reg, mu=0.9)

    loader = DatasetLoader(args.directory)
    train_dataset, test_dataset = loader.load(args.train_dataset_name, args.test_dataset_name)
    train_dataset = preprocess_dataset(train_dataset)
    train_labels, train_data = split_into_labels_and_data(train_dataset)

    test_dataset = preprocess_dataset(test_dataset)
    test_labels, test_data = split_into_labels_and_data(test_dataset)

    losses = []
    test_accuracies = []
    train_accuracies = []

    for i in range(50000):
        batch = sample(train_dataset, args.batch_size)
        label_batch, image_batch = split_into_labels_and_data(batch)

        model.set_mode('train')
        scores = model.forward(images=image_batch)
        losses.append(model.eval_loss(labels=label_batch, scores=scores))
        model.backward()
        optimizer.step()

        if i % 1000 == 0:
            model.set_mode('test')
            test_accuracy = model.test(test_labels, test_data)
            test_accuracies.append(test_accuracy)
            print('On iteration ' + str(i) + ' test accuracy: ', test_accuracy)

            train_accuracy = model.test(train_labels, train_data)
            train_accuracies.append(train_accuracy)
            print('On iteration ' + str(i) + ' train accuracy: ', train_accuracy)
            print('')

    plot(losses=losses, test_accuracies=test_accuracies)

    plt.plot(train_accuracies)
    plt.show()


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
