import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from neural_nets.dataset.DatasetLoader import DatasetLoader
from neural_nets.model.BatchNorm1D import BatchNorm1DTrain
from neural_nets.model.BatchNorm2D import BatchNorm2DTrain
from neural_nets.model.Conv2D import Conv2DTrain
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.model.Dropout1D import Dropout1DTrain
from neural_nets.model.Dropout2D import Dropout2DTrain
from neural_nets.model.Linear import LinearTrain
from neural_nets.model.Loss import Loss
from neural_nets.model.MaxPool import MaxPoolTrain
from neural_nets.model.Model import TrainModel
from neural_nets.model.Relu import ReluTrain
from neural_nets.optimizer.Optimizer import Optimizer
from neural_nets.optimizer.SGDNesterovMomentum import SGDNesterovMomentum
from neural_nets.utils.DatasetProcessingUtils import preprocess
from neural_nets.utils.DatasetProcessingUtils import sample
from neural_nets.utils.PlotUtils import plot


class Trainer:
    def __init__(self, layers: list, optimizer: Optimizer, loss_function: Loss):
        pass

    def train(self, num_epoch: int, train_labels: ndarray, train_data: ndarray, test_labels: ndarray,
              test_data: ndarray):
        pass


def run():
    # SGDMomentum.learning_rate = 0.0001

    SGDNesterovMomentum.learning_rate = 0.02
    SGDNesterovMomentum.mu = 0.9

    # RMSprop.learning_rate = 0.02
    # RMSprop.decay_rate = 0.9

    # Adam.learning_rate = 0.005
    # Adam.beta1 = 0.9
    # Adam.beta2 = 0.999

    # Adagrad.learning_rate = 0.02

    loss_function = CrossEntropyLoss()
    # loss = SVM_Loss(10.0)

    block_1 = 'conv1'
    l1 = Conv2DTrain.init(block_name=block_1,
                          num_filters=32,
                          filter_depth=1,
                          filter_height=3,
                          filter_width=3,
                          stride=1,
                          padding=1,
                          optimizer_class=SGDNesterovMomentum)

    l2 = BatchNorm2DTrain.init(block_name=block_1,
                               num_of_channels=32,
                               momentum=0.9,
                               optimizer_class=SGDNesterovMomentum)

    l3 = ReluTrain()
    l4 = Dropout2DTrain(keep_active_prob=0.9)
    l5 = MaxPoolTrain(pool_height=2, pool_width=2, stride=2)

    block_2 = 'conv2'
    l6 = Conv2DTrain.init(block_name=block_2,
                          num_filters=64,
                          filter_depth=32,
                          filter_height=3,
                          filter_width=3,
                          stride=1,
                          padding=1,
                          optimizer_class=SGDNesterovMomentum)

    l7 = BatchNorm2DTrain.init(block_name=block_2,
                               num_of_channels=64,
                               momentum=0.9,
                               optimizer_class=SGDNesterovMomentum)

    l8 = ReluTrain()
    l9 = Dropout2DTrain(keep_active_prob=0.7)
    l10 = MaxPoolTrain(pool_height=2, pool_width=2, stride=2)

    block_3 = 'linear1'
    l11 = LinearTrain.init(block_name=block_3,
                           input_dim=3136,
                           num_of_neurons=128,
                           optimizer_class=SGDNesterovMomentum)

    l12 = BatchNorm1DTrain.init(block_name=block_3,
                                input_dim=128,
                                momentum=0.9,
                                optimizer_class=SGDNesterovMomentum)

    l13 = ReluTrain()
    l14 = Dropout1DTrain(keep_active_prob=0.5)

    block_4 = 'linear2'
    l15 = LinearTrain.init(block_name=block_4,
                           input_dim=128,
                           num_of_neurons=10,
                           optimizer_class=SGDNesterovMomentum)

    train_layers = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15]
    train_model = TrainModel(layers=train_layers)

    # optimizer = SGDMomentum(model=train_model, learning_rate=0.0001, mu=0.9)
    # optimizer = SGDNesterovMomentum(model=train_model, learning_rate=0.02, mu=0.9)
    # optimizer = Adagrad(model=train_model, learning_rate=0.02)
    # optimizer = RMSprop(model=train_model, learning_rate=0.02, decay_rate=0.9)
    # optimizer = Adam(model=train_model, learning_rate=0.005, beta1=0.9, beta2=0.999)

    loader = DatasetLoader(r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv')

    train_labels, train_data, test_labels, test_data = loader.load('fmnist_train.npy', 'fmnist_test.npy')

    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    losses = []
    test_accuracies = []
    train_accuracies = []

    # model_forward_run = train_model.init_model()

    train_model, model_forward_run = train_model.load(r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv400.npz')

    batch_size = 64
    test_batch_size = 5000
    for i in range(1000):
        train_label_batch, train_image_batch = sample(labels=train_labels, data=train_data, batch_size=batch_size)
        train_image_batch = train_image_batch.reshape((batch_size, 1, 28, 28))

        model_forward_run, scores = train_model.forward(model_forward_run=model_forward_run, images=train_image_batch)
        data_loss, loss_run = loss_function.eval_data_loss(labels=train_label_batch, scores=scores)

        losses.append(data_loss)

        model_backward_run = train_model.backward(loss_function=loss_function,
                                                  model_forward_run=model_forward_run,
                                                  loss_run=loss_run)

        train_model = train_model.optimize(model_backward_run=model_backward_run)

        print(i)
        if i % 200 == 0:
            test_model = train_model.to_test(model_forward_run=model_forward_run)

            path = r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv'
            path += str(i)
            train_model.save(path=path, model_forward_run=model_forward_run)
            print('Saved model to: {}'.format(path))

            batch_test_accuracies = []
            for k in range(2):
                print(k)
                test_label_batch = test_labels[k * test_batch_size: k * test_batch_size + test_batch_size]
                test_image_batch = test_data[k * test_batch_size: k * test_batch_size + test_batch_size]

                test_image_batch = test_image_batch.reshape((test_batch_size, 1, 28, 28))
                batch_test_accuracies.append(test_model.test(labels=test_label_batch, images=test_image_batch))
            test_accuracy = np.mean(batch_test_accuracies)
            test_accuracies.append(test_accuracy)
            print('On iteration ' + str(i) + ' test accuracy: ', test_accuracy)

            batch_train_accuracies = []
            for k in range(12):
                print(k)
                train_label_batch = train_labels[k * test_batch_size: k * test_batch_size + test_batch_size]
                train_image_batch = train_data[k * test_batch_size: k * test_batch_size + test_batch_size]

                train_image_batch = train_image_batch.reshape((test_batch_size, 1, 28, 28))
                batch_train_accuracies.append(test_model.test(labels=train_label_batch, images=train_image_batch))
            train_accuracy = np.mean(batch_train_accuracies)
            train_accuracies.append(train_accuracy)
            print('On iteration ' + str(i) + ' train accuracy: ', train_accuracy)
            print('')

    plot(losses=losses, test_accuracies=test_accuracies)
    plt.plot(train_accuracies)
    plt.show()


if __name__ == "__main__":
    run()
