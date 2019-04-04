import matplotlib.pyplot as plt
import numpy as np

from neural_nets.dataset.DatasetLoader import DatasetLoader
from neural_nets.model.BatchNorm1D import BatchNorm1DTrain
from neural_nets.model.BatchNorm2D import BatchNorm2DTrain
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.model.Linear import LinearTrain
from neural_nets.model.Conv2D import Conv2DTrain
from neural_nets.model.MaxPool import MaxPoolTrain
from neural_nets.model.Dropout1D import Dropout2DTrain
from neural_nets.model.Model import TrainModel
from neural_nets.model.Optimizer import SGDNesterovMomentum, SGD, SGDMomentum
from neural_nets.model.Relu import ReluTrain
from neural_nets.utils.DatasetProcessingUtils import preprocess_dataset, sample, split_into_labels_and_data
from neural_nets.utils.PlotUtils import plot


def run():
    # linear_layer1 = LinearTrain(num_of_neurons=800, input_dim=784)
    # batch_norm1 = BatchNormTrain(input_dim=800, momentum=0.9)
    # relu_laye1 = ReluTrain()
    # linear_layer2 = LinearTrain(num_of_neurons=10, input_dim=800)

    loss_function = CrossEntropyLoss()
    # loss = SVM_Loss(10.0)

    l1 = Conv2DTrain(num_filters=8,
                     filter_depth=1,
                     filter_height=3,
                     filter_width=3,
                     stride=1,
                     padding=1)
    l2 = BatchNorm2DTrain(num_of_channels=8, momentum=0.9)
    l3 = ReluTrain()
    l4 = MaxPoolTrain(pool_height=2, pool_width=2, stride=2)

    l5 = Conv2DTrain(num_filters=16,
                     filter_depth=8,
                     filter_height=3,
                     filter_width=3,
                     stride=1,
                     padding=1)
    l6 = BatchNorm2DTrain(num_of_channels=16, momentum=0.9)
    l7 = ReluTrain()

    l8 = MaxPoolTrain(pool_height=2, pool_width=2, stride=2)

    l9 = LinearTrain(input_dim=784, num_of_neurons=128)
    l10 = BatchNorm1DTrain(input_dim=128, momentum=0.9)
    l11 = ReluTrain()

    l12 = Dropout2DTrain(keep_active_prob=0.5)
    l13 = LinearTrain(input_dim=128, num_of_neurons=10)

    train_model = TrainModel()
    train_model.add(layer=l1)
    train_model.add(layer=l2)
    train_model.add(layer=l3)
    train_model.add(layer=l4)
    train_model.add(layer=l5)
    train_model.add(layer=l6)
    train_model.add(layer=l7)
    train_model.add(layer=l8)
    train_model.add(layer=l9)
    train_model.add(layer=l10)
    train_model.add(layer=l11)
    train_model.add(layer=l12)
    train_model.add(layer=l13)

    # optimizer = SGD(model=train_model, learning_rate=0.001)
    # optimizer = SGDMomentum(model=train_model, learning_rate=0.0001, mu=0.9)
    optimizer = SGDNesterovMomentum(model=train_model, learning_rate=0.02, mu=0.9)

    loader = DatasetLoader(r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv')
    train_dataset, test_dataset = loader.load('mnist_train.csv', 'mnist_test.csv')
    train_dataset = preprocess_dataset(train_dataset)

    test_dataset = preprocess_dataset(test_dataset)
    test_labels, test_data = split_into_labels_and_data(test_dataset)
    num_test_samples = test_data.shape[0]
    test_data = test_data.reshape((num_test_samples, 1, 28, 28))

    losses = []
    test_accuracies = []
    train_accuracies = []

    test_model_params = train_model.init_test_model_params()

    batch_size = 64
    for i in range(5000):
        batch = sample(dataset=train_dataset, batch_size=batch_size)
        label_batch, image_batch = split_into_labels_and_data(batch)
        image_batch = image_batch.reshape((batch_size, 1, 28, 28))

        model_forward_run = train_model.forward(test_model_params=test_model_params, images=image_batch)
        data_loss, loss_run = loss_function.eval_data_loss(labels=label_batch, model_forward_run=model_forward_run)

        losses.append(data_loss)

        model_backward_run = optimizer.backward(loss_function=loss_function,
                                                model_forward_run=model_forward_run, loss_run=loss_run)

        optimizer.step(model_backward_run=model_backward_run)

        if i % 100 == 0:
            test_model = train_model.to_test(test_model_params)
            test_accuracy = test_model.test(labels=test_labels, images=test_data)
            test_accuracies.append(test_accuracy)
            print('On iteration ' + str(i) + ' test accuracy: ', test_accuracy)

            train_batch = sample(dataset=train_dataset, batch_size=num_test_samples)
            train_labels, train_data = split_into_labels_and_data(train_batch)
            train_data = train_data.reshape((num_test_samples, 1, 28, 28))
            train_accuracy = test_model.test(labels=train_labels, images=train_data)
            train_accuracies.append(train_accuracy)
            print('On iteration ' + str(i) + ' train accuracy: ', train_accuracy)
            print('')

    plot(losses=losses, test_accuracies=test_accuracies)

    plt.plot(train_accuracies)
    plt.show()


if __name__ == "__main__":
    run()
