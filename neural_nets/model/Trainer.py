import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from neural_nets.model.BatchNorm1D import BatchNorm1DTrain
from neural_nets.model.BatchNorm2D import BatchNorm2DTrain
from neural_nets.model.Conv2D import Conv2DTrain
from neural_nets.model.Dropout1D import Dropout1DTrain
from neural_nets.model.Dropout2D import Dropout2DTrain
from neural_nets.model.Linear import LinearTrain
from neural_nets.model.Loss import Loss
from neural_nets.model.MaxPool import MaxPoolTrain
from neural_nets.model.Model import TrainModel
from neural_nets.model.Relu import ReluTrain
from neural_nets.utils.DatasetProcessingUtils import sample
from neural_nets.utils.PlotUtils import plot


class Trainer:
    def __init__(self, optimizer, loss_function: Loss):
        self.optimizer = optimizer
        self.layers = []
        self.loss_function = loss_function
        self.train_model = TrainModel(layers=[])
        self.model_forward_run = []

    def add_linear(self, block_name: str, input_dim: int, num_of_neurons: int):
        """
        Adds linear layer.
        :param block_name: is a name used to generate key for model saving.
        :param input_dim: is a number of incoming features to the layer.
        :param num_of_neurons: is a number of neurons in a layer.
        :return:
        """
        self.layers.append(LinearTrain.init(block_name=block_name,
                                            input_dim=input_dim,
                                            num_of_neurons=num_of_neurons,
                                            optimizer_class=self.optimizer))

    def add_conv2d(self, block_name: str, num_filters: int, filter_depth: int, filter_height: int, filter_width: int,
                   stride: int, padding: int):
        """
        Adds Convolutional 2D layer.
        :param block_name: is a name used to generate key for model saving.
        :param num_filters: is a number of filters (kernels).
        :param filter_depth: is a spatial depth of a single filter.
        :param filter_height: is a spatial height of a filter.
        :param filter_width: is a spatial width of a filter.
        :param stride: is a parameter with which we slide the filter.
        :param padding: is a number of zero pads around the input border.
        :return:
        """
        self.layers.append(Conv2DTrain.init(block_name=block_name,
                                            num_filters=num_filters,
                                            filter_depth=filter_depth,
                                            filter_height=filter_height,
                                            filter_width=filter_width,
                                            stride=stride,
                                            padding=padding,
                                            optimizer_class=self.optimizer))

    def add_batch_norm2d(self, block_name: str, num_of_channels: int, momentum: float):
        """
        Adds Batch Normalization spatial layer.
        :param block_name: is a name used to generate key for model saving.
        :param num_of_channels: is a spatial depth of an input.
        :param momentum: is a forgetting constant.
        :return:
        """
        self.layers.append(BatchNorm2DTrain.init(block_name=block_name,
                                                 num_of_channels=num_of_channels,
                                                 momentum=momentum,
                                                 optimizer_class=self.optimizer))

    def add_batch_norm1d(self, block_name: str, input_dim: int, momentum: float):
        """
        Adds Batch Normalization layer.
        :param block_name: is a name used to generate key for model saving.
        :param input_dim: is a number of input features.
        :param momentum: is a forgetting constant.
        :return:
        """
        self.layers.append(BatchNorm1DTrain.init(block_name=block_name,
                                                 input_dim=input_dim,
                                                 momentum=momentum,
                                                 optimizer_class=self.optimizer))

    def add_relu(self):
        """
        Adds ReLU activation function layer.
        :return:
        """
        self.layers.append(ReluTrain())

    def add_dropout1d(self, keep_active_prob: float):
        """
        Adds dropout layer.
        :param keep_active_prob: is a probability to keep activations active.
        :return:
        """
        self.layers.append(Dropout1DTrain(keep_active_prob=keep_active_prob))

    def add_dropout2d(self, keep_active_prob: float):
        """
        Adds spatial dropout layer.
        :param keep_active_prob: is a probability to keep activations active.
        :return:
        """
        self.layers.append(Dropout2DTrain(keep_active_prob=keep_active_prob))

    def add_max_pool(self, pool_height: int, pool_width: int, stride: int):
        """
        Adds Max Pooling layer.
        :param pool_height: is a spatial height of a pool window.
        :param pool_width: is a spatial width of a pool window.
        :param stride: is a parameter with which we slide the window.
        :return:
        """
        self.layers.append(MaxPoolTrain(pool_height=pool_height,
                                        pool_width=pool_width,
                                        stride=stride))

    def compile(self):
        """
        Compiles the model after all layers have been added.
        :return:
        """
        self.train_model = TrainModel(layers=self.layers)
        self.model_forward_run = self.train_model.init_model()

    def load_model(self, path: str):
        """
        Loads model weights from a specified file with .npy extension.
        :param path: is a weights file path.
        :return:
        """
        self.train_model, self.model_forward_run = self.train_model.load(path)

    def train(self, num_epoch: int, batch_size: int, test_batch_size: int, dataset: tuple, image_shape: tuple,
              snapshot: int, snapshot_dir=''):
        """
        Trains model with a specified number of epochs.
        :param num_epoch: is a number of training epochs.
        :param batch_size: is a train batch size.
        :param test_batch_size: is a test batch size.
        :param dataset: is a tuple of train labels, train data, test labels and test data.
        :param image_shape: is a tuple of shape of a single image. Should be in a format C x H x W,
        where C is a number of channels, H is an image height and W is an image width.
        :param snapshot: is number of epochs after the next model dump done.
        :param snapshot_dir: is a directory to save snapshot files.
        :return:
        """

        train_labels, train_data, test_labels, test_data = dataset
        C, H, W = image_shape

        losses = []
        test_accuracies = []
        train_accuracies = []
        for i in range(num_epoch):
            train_label_batch, train_image_batch = sample(labels=train_labels, data=train_data, batch_size=batch_size)
            train_image_batch = train_image_batch.reshape((batch_size, C, H, W))

            self.model_forward_run, scores = self.train_model.forward(model_forward_run=self.model_forward_run,
                                                                      images=train_image_batch)
            data_loss, loss_run = self.loss_function.eval_data_loss(labels=train_label_batch, scores=scores)

            losses.append(data_loss)

            model_backward_run = self.train_model.backward(loss_function=self.loss_function,
                                                           model_forward_run=self.model_forward_run,
                                                           loss_run=loss_run)

            self.train_model = self.train_model.optimize(model_backward_run=model_backward_run)

            print(i)
            if i % snapshot == 0:
                test_model = self.train_model.to_test(model_forward_run=self.model_forward_run)

                path = snapshot_dir
                path += str(i)
                self.train_model.save(path=path, model_forward_run=self.model_forward_run)
                print('Saved model to: {}'.format(path))

                batch_test_accuracies = []
                for k in range(int(test_labels.size / test_batch_size)):
                    print(k)
                    test_label_batch = test_labels[k * test_batch_size: k * test_batch_size + test_batch_size]
                    test_image_batch = test_data[k * test_batch_size: k * test_batch_size + test_batch_size]

                    test_image_batch = test_image_batch.reshape((test_batch_size, C, H, W))
                    batch_test_accuracies.append(test_model.test(labels=test_label_batch, images=test_image_batch))
                test_accuracy = np.mean(batch_test_accuracies)
                test_accuracies.append(test_accuracy)
                print('On iteration ' + str(i) + ' test accuracy: ', test_accuracy)

                batch_train_accuracies = []
                for k in range(int(train_labels.size / test_batch_size)):
                    print(k)
                    train_label_batch = train_labels[k * test_batch_size: k * test_batch_size + test_batch_size]
                    train_image_batch = train_data[k * test_batch_size: k * test_batch_size + test_batch_size]

                    train_image_batch = train_image_batch.reshape((test_batch_size, C, H, W))
                    batch_train_accuracies.append(test_model.test(labels=train_label_batch, images=train_image_batch))
                train_accuracy = np.mean(batch_train_accuracies)
                train_accuracies.append(train_accuracy)
                print('On iteration ' + str(i) + ' train accuracy: ', train_accuracy)
                print('')

        plot(losses=losses, test_accuracies=test_accuracies)
        plt.plot(train_accuracies)
        plt.show()

    def predict(self, image: ndarray):
        """
        Plow an input image and prints out the predict class into console.
        :param image: is an image to predict class on.
        :return:
        """
        plt.matshow(image[0])
        plt.show()

        test_model = self.train_model.to_test(model_forward_run=self.model_forward_run)
        result = test_model.predict(image=image)
        return result
