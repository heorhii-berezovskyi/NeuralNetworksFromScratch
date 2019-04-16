import matplotlib.pyplot as plt
import numpy as np

from neural_nets.dataset.DatasetLoader import DatasetLoader
from neural_nets.model.BatchNorm1D import BatchNorm1DTrain
from neural_nets.model.BatchNorm2D import BatchNorm2DTrain
from neural_nets.model.Conv2D import Conv2DTrain
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.model.Dropout1D import Dropout1DTrain
from neural_nets.model.Dropout2D import Dropout2DTrain
from neural_nets.model.Linear import LinearTrain
from neural_nets.model.MaxPool import MaxPoolTrain
from neural_nets.model.Model import TrainModel
from neural_nets.model.Relu import ReluTrain
from neural_nets.optimizer.RMSprop import RMSprop
from neural_nets.utils.DatasetProcessingUtils import preprocess_dataset, sample, split_into_labels_and_data
from neural_nets.utils.PlotUtils import plot


def run():
    learning_rate = 0.001
    mu = 0.9
    # SGDNesterovMomentum.learning_rate = learning_rate
    # SGDNesterovMomentum.mu = mu

    RMSprop.learning_rate = 0.02
    RMSprop.decay_rate = 0.9

    loss_function = CrossEntropyLoss()
    # loss = SVM_Loss(10.0)

    weights1 = Conv2DTrain.init_weights(num_filters=32,
                                        filter_depth=1,
                                        filter_height=3,
                                        filter_width=3)
    l1 = Conv2DTrain(weights=weights1,
                     stride=1,
                     padding=1,
                     optimizer=RMSprop.init_memory(weights=weights1))

    weights2 = BatchNorm2DTrain.init_weights(num_of_channels=32)
    l2 = BatchNorm2DTrain(weights=weights2,
                          momentum=0.9,
                          optimizer=RMSprop.init_memory(weights=weights2))
    l3 = ReluTrain()
    l4 = Dropout2DTrain(keep_active_prob=0.9)
    l5 = MaxPoolTrain(pool_height=2, pool_width=2, stride=2)

    weights6 = Conv2DTrain.init_weights(num_filters=64,
                                        filter_depth=32,
                                        filter_height=3,
                                        filter_width=3)
    l6 = Conv2DTrain(weights=weights6,
                     stride=1,
                     padding=1,
                     optimizer=RMSprop.init_memory(weights=weights6))
    weights7 = BatchNorm2DTrain.init_weights(num_of_channels=64)
    l7 = BatchNorm2DTrain(weights=weights7,
                          momentum=0.9,
                          optimizer=RMSprop.init_memory(weights=weights7))
    l8 = ReluTrain()
    l9 = Dropout2DTrain(keep_active_prob=0.7)
    l10 = MaxPoolTrain(pool_height=2, pool_width=2, stride=2)

    weights11 = LinearTrain.init_weights(input_dim=3136, num_of_neurons=128)
    l11 = LinearTrain(weights=weights11,
                      optimizer=RMSprop.init_memory(weights=weights11))

    weights12 = BatchNorm1DTrain.init_weights(input_dim=128)
    l12 = BatchNorm1DTrain(weights=weights12,
                           momentum=0.9,
                           optimizer=RMSprop.init_memory(weights=weights12))
    l13 = ReluTrain()
    l14 = Dropout1DTrain(keep_active_prob=0.5)

    weights15 = LinearTrain.init_weights(input_dim=128, num_of_neurons=10)
    l15 = LinearTrain(weights=weights15,
                      optimizer=RMSprop.init_memory(weights=weights15))

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
    train_model.add(layer=l14)
    train_model.add(layer=l15)

    # optimizer = SGDMomentum(model=train_model, learning_rate=0.0001, mu=0.9)
    # optimizer = SGDNesterovMomentum(model=train_model, learning_rate=0.02, mu=0.9)
    # optimizer = Adagrad(model=train_model, learning_rate=0.02)
    # optimizer = RMSprop(model=train_model, learning_rate=0.02, decay_rate=0.9)
    # optimizer = Adam(model=train_model, learning_rate=np.float64(0.005), beta1=np.float64(0.9), beta2=np.float64(0.999))

    loader = DatasetLoader(r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv')

    train_dataset, test_dataset = loader.load('fmnist_train.npy', 'fmnist_test.npy')
    train_dataset = preprocess_dataset(train_dataset)
    test_dataset = preprocess_dataset(test_dataset)

    losses = []
    test_accuracies = []
    train_accuracies = []

    model_forward_run = train_model.init_model()

    batch_size = 64
    test_batch_size = 5000
    for i in range(1000):
        batch = sample(dataset=train_dataset, batch_size=batch_size)
        label_batch, image_batch = split_into_labels_and_data(batch)
        image_batch = image_batch.reshape((batch_size, 1, 28, 28))

        model_forward_run, scores = train_model.forward(model_forward_run=model_forward_run, images=image_batch)
        data_loss, loss_run = loss_function.eval_data_loss(labels=label_batch, scores=scores)

        losses.append(data_loss)

        model_backward_run = train_model.backward(loss_function=loss_function,
                                                  model_forward_run=model_forward_run,
                                                  loss_run=loss_run)

        train_model = train_model.optimize(model_backward_run=model_backward_run)

        print(i)
        if i % 200 == 0:
            test_model = train_model.to_test(model_forward_run)

            # path = r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv400.npz'
            # test_model.load(path=path)

            path = r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv'
            path += str(i)
            test_model.save(path)
            print('Saved model to: {}'.format(path))

            batch_test_accuracies = []
            for k in range(4):
                print(k)
                test_batch = sample(dataset=test_dataset, batch_size=test_batch_size)
                test_labels, test_data = split_into_labels_and_data(test_batch)
                test_data = test_data.reshape((test_batch_size, 1, 28, 28))
                batch_test_accuracies.append(test_model.test(labels=test_labels, images=test_data))
            test_accuracy = np.mean(batch_test_accuracies)
            test_accuracies.append(test_accuracy)
            print('On iteration ' + str(i) + ' test accuracy: ', test_accuracy)

            batch_train_accuracies = []
            for k in range(24):
                print(k)
                train_batch = sample(dataset=train_dataset, batch_size=test_batch_size)
                train_labels, train_data = split_into_labels_and_data(data=train_batch)
                train_data = train_data.reshape((test_batch_size, 1, 28, 28))
                batch_train_accuracies.append(test_model.test(labels=train_labels, images=train_data))
            train_accuracy = np.mean(batch_train_accuracies)
            train_accuracies.append(train_accuracy)
            print('On iteration ' + str(i) + ' train accuracy: ', train_accuracy)
            print('')

    plot(losses=losses, test_accuracies=test_accuracies)
    plt.plot(train_accuracies)
    plt.show()


if __name__ == "__main__":
    run()
