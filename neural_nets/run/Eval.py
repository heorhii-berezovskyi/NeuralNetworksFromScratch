import argparse

import numpy as np

from neural_nets.dataset.DatasetLoader import DatasetLoader
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.run.ModelSelector import ModelSelector


def run(args):
    train_model, model_forward_run, train_mean = ModelSelector().select(args.model).load(path=args.snapshot_from)
    print('Loaded model from ' + args.snapshot_from)
    test_model = train_model.to_test(model_forward_run)

    loader = DatasetLoader()
    test_labels, test_data = loader.load(path=args.test_dataset_path)
    test_data -= train_mean

    loss_function = CrossEntropyLoss()

    test_batch_size = args.test_batch_size
    batch_test_accuracies = []
    batch_test_loss = []
    C, H, W = args.num_of_channels, args.image_size, args.image_size
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
    print('Test accuracy:', test_accuracy)

    test_loss = np.mean(batch_test_loss)
    print('Test loss:', test_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluates specified model from scratch with specified parameters.')
    parser.add_argument("--snapshot_from", type=str, help="Path to snapshot to continue training from.",
                        default=r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv\model\convnet400.npz')

    parser.add_argument("--model", type=str, help="Model name.", default='convnet')
    parser.add_argument("--test_dataset_path", type=str, help="Test data set path with .npy format.",
                        default=r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv\fmnist_test.npy')

    parser.add_argument("--test_batch_size", type=int, help="Number of samples in test batch.", default=5000)

    parser.add_argument("--num_of_channels", type=int, help="Number of channels in a single image.", default=1)
    parser.add_argument("--image_size", type=int, help="Dimension of a square image.", default=28)

    _args = parser.parse_args()
    run(_args)
