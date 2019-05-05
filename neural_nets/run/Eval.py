import argparse

import numpy as np

from neural_nets.dataset.DatasetLoader import DatasetLoader
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.model.Model import TestModel
from neural_nets.run.ModelSelector import ModelSelector


def run(args):
    test_model, train_mean = TestModel.load(path=args.snapshot_from,
                                            train_model=ModelSelector().select(name=args.model))
    print('Loaded model from ' + args.snapshot_from)

    test_dataset_loader = DatasetLoader(dataset_directory=args.test_dataset_dir)
    test_labels, test_data = test_dataset_loader.load(type='test')
    test_data -= train_mean

    loss_function = CrossEntropyLoss()

    test_batch_size = args.test_batch_size
    batch_test_accuracies = []
    batch_test_loss = []
    for k in range(int(test_labels.size / test_batch_size)):
        test_label_batch = test_labels[k * test_batch_size: k * test_batch_size + test_batch_size]
        test_image_batch = test_data[k * test_batch_size: k * test_batch_size + test_batch_size]

        scores = test_model.eval_scores(test_image_batch)
        accuracy = test_model.eval_accuracy(test_label_batch, scores)
        data_loss, _ = loss_function.eval_data_loss(test_label_batch, scores)

        batch_test_accuracies.append(accuracy)
        batch_test_loss.append(data_loss)

    test_accuracy = np.mean(batch_test_accuracies)
    print('Test accuracy:', test_accuracy)

    test_loss = np.mean(batch_test_loss)
    print('Test loss:', test_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates specified model from scratch with specified parameters.')
    parser.add_argument("--snapshot_from", type=str, help="Path to snapshot to continue training from.",
                        default=r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv\model\convnet400.npz')

    parser.add_argument("--model", type=str, help="Model name.", default='convnet')
    parser.add_argument("--test_dataset_dir", type=str,
                        help="Test data set directory with labels and data in .npy format.",
                        default=r'C:\Users\heorhii.berezovskyi\Documents\mnist')

    parser.add_argument("--test_batch_size", type=int, help="Number of samples in test batch.", default=5000)

    _args = parser.parse_args()
    run(_args)
