import argparse

import numpy as np

from neural_nets.dataset.DatasetLoader import DatasetLoader
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.utils.DatasetProcessingUtils import remove_mean_on_train, remove_mean_on_test
from neural_nets.run.ModelSelector import ModelSelector


def run(args):
    loader = DatasetLoader(directory=args.dataset_dir)
    train_labels, train_data = loader.load(dataset_name='fmnist_train.npy')
    test_labels, test_data = loader.load(dataset_name='fmnist_test.npy')

    train_data, train_mean = remove_mean_on_train(train_data=train_data)
    test_data = remove_mean_on_test(test_data=test_data, train_mean=train_mean)

    train_model, model_forward_run = ModelSelector().select(args.model).load(path=args.snapshot_from)
    print('Loaded model from ' + args.snapshot_from)
    test_model = train_model.to_test(model_forward_run)

    loss_function = CrossEntropyLoss()

    test_batch_size = 5000
    batch_test_accuracies = []
    batch_test_loss = []
    C, H, W = 1, 28, 28
    for k in range(int(test_labels.size / test_batch_size)):
        test_label_batch = test_labels[k * test_batch_size: k * test_batch_size + test_batch_size]
        test_image_batch = test_data[k * test_batch_size: k * test_batch_size + test_batch_size]

        test_image_batch = test_image_batch.reshape((test_batch_size, C, H, W))

        scores = test_model.eval_scores(test_image_batch)
        accuracy = test_model.eval_accuracy(test_label_batch, scores)
        data_loss, _ = loss_function.eval_data_loss(test_label_batch, scores)

        batch_test_accuracies.append(accuracy)
        batch_test_loss.append(data_loss)

    test_accuracy = np.mean(batch_test_accuracies)
    print(test_accuracy)

    test_loss = np.mean(batch_test_loss)
    print(test_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains specified model from scratch with specified parameters.')
    parser.add_argument("--snapshot_from", type=str, help="Path to snapshot to continue training from.",
                        default=r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv600.npz')

    parser.add_argument("--model", type=str, help="Model name.", default='convnet')
    parser.add_argument("--dataset_dir", type=str, help="Path to directory with data set.",
                        default=r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv')

    _args = parser.parse_args()
    run(_args)
