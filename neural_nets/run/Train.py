import argparse

from neural_nets.dataset.DatasetLoader import DatasetLoader
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.optimizer.Adam import Adam
from neural_nets.run.ModelSelector import ModelSelector
from neural_nets.run.Trainer import Trainer
from neural_nets.utils.DatasetProcessingUtils import remove_mean_on_train, remove_mean_on_test
from neural_nets.utils.OptimizerSelector import OptimizerSelector


def run(args):
    loader = DatasetLoader()
    train_labels, train_data = loader.load(path=args.train_dataset_path)
    test_labels, test_data = loader.load(path=args.test_dataset_path)

    train_data, train_mean = remove_mean_on_train(train_data=train_data)
    test_data = remove_mean_on_test(test_data=test_data, train_mean=train_mean)

    if args.optimizer != '':
        optimizer = OptimizerSelector().select(args.optimizer)
        optimizer.learning_rate = args.learning_rate
    else:
        optimizer = Adam

    if args.snapshot_from != '':
        train_model, model_forward_run, loaded_train_mean = ModelSelector().select(args.model).load(
            path=args.snapshot_from)
        print('Loaded model from ' + args.snapshot_from)
        if args.optimizer != '':
            train_model = train_model.with_optimizer(args.optimizer)
        trainer = Trainer(train_model=train_model,
                          model_forward_run=model_forward_run,
                          loss_function=CrossEntropyLoss(),
                          train_mean=train_mean)
    else:
        train_model = ModelSelector().select(args.model).with_optimizer(optimizer)
        trainer = Trainer(train_model=train_model,
                          model_forward_run=[],
                          loss_function=CrossEntropyLoss(),
                          train_mean=train_mean)
        trainer.compile()

    trainer.train(num_iters=args.num_iters,
                  batch_size=args.batch_size,
                  test_batch_size=args.test_batch_size,
                  dataset=(train_labels, train_data, test_labels, test_data),
                  image_shape=(args.num_of_channels, args.image_size, args.image_size),
                  snapshot=args.snapshot,
                  snapshot_dir=args.snapshot_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains specified model with specified parameters.')
    parser.add_argument("--train_dataset_path", type=str, help="Train data set path with .npy format.",
                        default=r'C:\Users\ГеоргійБерезовський\Documents\NN_from_scratch\dataset\fmnist_train.npy')

    parser.add_argument("--test_dataset_path", type=str, help="Test data set path with .npy format.",
                        default=r'C:\Users\ГеоргійБерезовський\Documents\NN_from_scratch\dataset\fmnist_test.npy')

    parser.add_argument("--num_of_channels", type=int, help="Number of channels in a single image.", default=1)
    parser.add_argument("--image_size", type=int, help="Dimension of a square image.", default=28)

    parser.add_argument("--snapshot_from", type=str, help="Path to snapshot to continue training from.",
                        default=r'C:\Users\ГеоргійБерезовський\Documents\NN_from_scratch\models\convnet400.npz')
    parser.add_argument("--optimizer", type=str, help="Optimizer name.", default='')
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=0.002)
    parser.add_argument("--model", type=str, help="Model name.", default='convnet')

    parser.add_argument("--test_batch_size", type=int, help="Number of samples in test batch.", default=5000)
    parser.add_argument("--num_iters", type=int, help="Number of training iterations.", default=1000)
    parser.add_argument("--batch_size", type=int, help="Training batch size.", default=64)
    parser.add_argument("--snapshot", type=int, help="Number of iterations after the next snapshot done.", default=100)
    parser.add_argument("--snapshot_dir", type=str, help="Path to a directory to save snapshots.",
                        default=r'C:\Users\ГеоргійБерезовський\Documents\NN_from_scratch\models\\convnet')

    _args = parser.parse_args()
    run(_args)
