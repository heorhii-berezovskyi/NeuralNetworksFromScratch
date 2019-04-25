import matplotlib.pyplot as plt
import numpy as np

from neural_nets.model.Loss import Loss
from neural_nets.model.Model import TrainModel
from neural_nets.utils.DatasetProcessingUtils import sample
from neural_nets.utils.PlotUtils import plot
from neural_nets.utils.DatasetProcessingUtils import noise


class Trainer:
    def __init__(self, train_model: TrainModel, model_forward_run: list, train_mean: float, loss_function: Loss):
        """
        :param loss_function: is a specified loss function.
        """
        self.train_model = train_model
        self.loss_function = loss_function
        self.model_forward_run = model_forward_run
        self.train_mean = train_mean

    def compile(self):
        """
        Compiles the model after all layers have been added.
        :return:
        """
        self.model_forward_run = self.train_model.init_model()
        np.save('losses.npy', np.array([0], dtype=float))
        np.save('accuracies.npy', np.array([0], dtype=float))

    def load_model(self, path: str):
        """
        Loads model weights from a specified file with .npy extension.
        :param path: is a weights file path.
        :return:
        """
        self.train_model, self.model_forward_run, train_mean = self.train_model.load(path)
        self.train_mean = train_mean[0]

    def train(self, num_iters: int, batch_size: int, test_batch_size: int, dataset: tuple, image_shape: tuple,
              snapshot: int, snapshot_dir: str):
        """
        Trains model with a specified number of epochs.
        :param num_iters: is a number of training iterations.
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
        for i in range(num_iters):
            train_label_batch, train_image_batch = sample(labels=train_labels, data=train_data, batch_size=batch_size)
            train_image_batch = train_image_batch.reshape((batch_size, C, H, W))

            for m in range(batch_size):
                train_image_batch[m] = noise(train_image_batch[m])

            self.model_forward_run, scores = self.train_model.forward(model_forward_run=self.model_forward_run,
                                                                      images=train_image_batch)
            data_loss, loss_run = self.loss_function.eval_data_loss(labels=train_label_batch, scores=scores)
            print('On iter', i, 'loss', data_loss)

            losses.append(data_loss)

            model_backward_run = self.train_model.backward(loss_function=self.loss_function,
                                                           model_forward_run=self.model_forward_run,
                                                           loss_run=loss_run)

            self.train_model = self.train_model.optimize(model_backward_run=model_backward_run)

            if i % snapshot == 0:
                test_model = self.train_model.to_test(model_forward_run=self.model_forward_run)

                path = snapshot_dir + str(i)
                self.train_model.save(path=path, model_forward_run=self.model_forward_run, train_mean=self.train_mean)
                print('Saved model to: {}'.format(path))

                batch_test_accuracies = []
                for k in range(int(test_labels.size / test_batch_size)):
                    test_label_batch = test_labels[k * test_batch_size: k * test_batch_size + test_batch_size]
                    test_image_batch = test_data[k * test_batch_size: k * test_batch_size + test_batch_size]

                    test_image_batch = test_image_batch.reshape((test_batch_size, C, H, W))
                    scores = test_model.eval_scores(images=test_image_batch)
                    batch_test_accuracies.append(
                        test_model.eval_accuracy(labels=test_label_batch, scores=scores))
                test_accuracy = np.mean(batch_test_accuracies)
                test_accuracies.append(test_accuracy)
                print('On iteration ' + str(i) + ' test accuracy: ', test_accuracy)

                loss = np.load(file='losses.npy')
                acc = np.load(file='accuracies.npy')
                new_loss = np.append(loss, data_loss)
                new_acc = np.append(acc, test_accuracy)

                np.save('losses.npy', new_loss)
                np.save('accuracies.npy', new_acc)

        plot(losses=losses, test_accuracies=test_accuracies)
        plt.show()
