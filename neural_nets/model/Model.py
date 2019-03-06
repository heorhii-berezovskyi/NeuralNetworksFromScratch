import argparse
from collections import deque

from numpy import argmax
from numpy import mean
from numpy import ndarray

from neural_nets.dataset.DatasetLoader import DatasetLoader
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.model.Layer import Layer
from neural_nets.model.Linear import Linear
from neural_nets.model.Relu import Relu
from neural_nets.model.Visitor import GradientUpdateVisitor
from neural_nets.model.Visitor import RegularizationVisitor
from neural_nets.utils.DatasetProcessingUtils import preprocess_dataset
from neural_nets.utils.DatasetProcessingUtils import sample
from neural_nets.utils.DatasetProcessingUtils import split_into_labels_and_data
from neural_nets.utils.PlotUtils import plot


class Model:
    """
    Neural net model representative class. Model can be comprised of different type, size and count of layers.
    """

    def __init__(self, reg: float):
        self.layers = []
        self.reg = reg
        self.gradients = deque()

    def add_layer(self, layer: Layer):
        """
        Adds layer to layers list.
        :param layer: is a model layer representative.
        """
        self.layers.append(layer)

    def forward(self, labels: ndarray, images: ndarray):
        """
        Computes forward propagation on each layer.
        :param labels: is a training labels.
        :param images: is a training images.
        :return: model loss.
        """
        visitor = RegularizationVisitor(reg_strength=self.reg)
        input_data = images.T
        for layer in self.layers[:-1]:
            output_data = layer.forward(input_data)
            layer.accept(visitor)
            input_data = output_data
        data_loss = self.layers[-1].forward((labels, input_data))
        reg_loss = visitor.get_reg_loss()
        loss = data_loss + reg_loss
        return loss

    def backward(self):
        """
        Performs backward propagation and gathers gradients of each layer by previous layer.
        """
        for layer in reversed(self.layers):
            self.gradients.append(layer.backward())

    def train_on_batch(self, lr: float):
        """
        Updates weights of model layers.
        :param lr: is a specified learning rate.
        """
        dL_hidden = self.gradients.popleft()
        for layer in reversed(self.layers[:-1]):
            visitor = GradientUpdateVisitor(loss_grad=dL_hidden, updating_loss_grad_term=self.gradients.popleft(),
                                            reg_strength=self.reg, lr=lr)
            layer.accept(visitor)
            dL_hidden = visitor.get_updated_loss_gradient()

    def test(self, labels: ndarray, images: ndarray):
        """
        Evaluates accuracy of a trained model on test labels and data.
        :param labels: is a batch of labels.
        :param images: is a batch of images.
        """
        input_data = images.T
        for layer in self.layers[:-1]:
            output_data = layer.forward(input_data)
            input_data = output_data
        predicted_class = argmax(input_data, axis=0)
        accuracy = mean(predicted_class == labels)
        return accuracy


def run(args):
    linear_layer1 = Linear(args.num_of_hidden_neurons, (args.size_of_input_vector, args.batch_size))
    relu_layer = Relu((args.num_of_hidden_neurons, args.batch_size))
    linear_layer2 = Linear(args.num_of_classes, (args.num_of_hidden_neurons, args.batch_size))
    loss_layer = CrossEntropyLoss((args.num_of_classes, args.batch_size))

    model = Model(reg=args.reg)
    model.add_layer(linear_layer1)
    model.add_layer(relu_layer)
    model.add_layer(linear_layer2)
    model.add_layer(loss_layer)

    loader = DatasetLoader(args.directory)
    train_dataset, test_dataset = loader.load(args.train_dataset_name, args.test_dataset_name)
    train_dataset = preprocess_dataset(train_dataset)

    test_dataset = preprocess_dataset(test_dataset)
    test_labels, test_data = split_into_labels_and_data(test_dataset)

    losses = []
    test_accuracies = []

    learning_rates = [0.0001, 0.0001 / 2.0, 0.0001 / 4.0, 0.0001 / 8.0, 0.0001 / 16.0]
    for lr in learning_rates:
        for i in range(10000):
            batch = sample(train_dataset, args.batch_size)
            label_batch, image_batch = split_into_labels_and_data(batch)
            losses.append(model.forward(label_batch, image_batch))
            model.backward()
            model.train_on_batch(lr)
        test_accuracies.append(model.test(test_labels, test_data))

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
