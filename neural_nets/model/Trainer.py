import matplotlib.pyplot as plt

from neural_nets.dataset.DatasetLoader import DatasetLoader
from neural_nets.model.BatchNorm import BatchNormTrain
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.model.Linear import LinearTrain
from neural_nets.model.Model import TrainModel
from neural_nets.model.Optimizer import SGDNesterovMomentum
from neural_nets.model.Relu import ReluTrain
from neural_nets.utils.DatasetProcessingUtils import preprocess_dataset, sample, split_into_labels_and_data
from neural_nets.utils.PlotUtils import plot


def run():
    linear_layer1 = LinearTrain(num_of_neurons=800, input_dim=784)
    batch_norm1 = BatchNormTrain(input_dim=800, momentum=0.9)
    relu_laye1 = ReluTrain()
    linear_layer2 = LinearTrain(num_of_neurons=10, input_dim=800)

    loss_function = CrossEntropyLoss()
    # loss = SVM_Loss(10.0)

    train_model = TrainModel()
    train_model.add(linear_layer1)
    train_model.add(batch_norm1)
    train_model.add(relu_laye1)
    train_model.add(linear_layer2)

    # optimizer = SGD( model=train_model, learning_rate=0.01)
    # optimizer = SGDMomentum(model=train_model, learning_rate=0.01, mu=0.9)
    optimizer = SGDNesterovMomentum(model=train_model, learning_rate=0.01, mu=0.9)

    loader = DatasetLoader(r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv')
    train_dataset, test_dataset = loader.load('mnist_train.csv', 'mnist_test.csv')
    train_dataset = preprocess_dataset(train_dataset)
    train_labels, train_data = split_into_labels_and_data(train_dataset)

    test_dataset = preprocess_dataset(test_dataset)
    test_labels, test_data = split_into_labels_and_data(test_dataset)

    losses = []
    test_accuracies = []
    train_accuracies = []

    test_model_params = train_model.init_test_model_params()

    for i in range(50000):
        batch = sample(dataset=train_dataset, batch_size=64)
        label_batch, image_batch = split_into_labels_and_data(batch)

        model_forward_run = train_model.forward(test_model_params=test_model_params, images=image_batch)
        data_loss, loss_run = loss_function.eval_data_loss(labels=label_batch, model_forward_run=model_forward_run)

        losses.append(data_loss)

        model_backward_run = optimizer.backward(loss_function=loss_function,
                                                model_forward_run=model_forward_run, loss_run=loss_run)

        optimizer.step(model_backward_run=model_backward_run)

        if i % 1000 == 0:
            test_model = train_model.to_test(test_model_params)
            test_accuracy = test_model.test(labels=test_labels, images=test_data)
            test_accuracies.append(test_accuracy)
            print('On iteration ' + str(i) + ' test accuracy: ', test_accuracy)

            train_accuracy = test_model.test(labels=train_labels, images=train_data)
            train_accuracies.append(train_accuracy)
            print('On iteration ' + str(i) + ' train accuracy: ', train_accuracy)
            print('')

    plot(losses=losses, test_accuracies=test_accuracies)

    plt.plot(train_accuracies)
    plt.show()


if __name__ == "__main__":
    run()