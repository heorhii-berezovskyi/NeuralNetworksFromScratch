import argparse

from neural_nets.dataset.DatasetLoader import DatasetLoader
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.optimizer.Adam import Adam
from neural_nets.run.Trainer import Trainer
from neural_nets.utils.DatasetProcessingUtils import remove_mean_on_train, remove_mean_on_test
from neural_nets.run.ModelSelector import ModelSelector
from neural_nets.utils.OptimizerSelector import OptimizerSelector


def run(args):
    if args.optimizer != '':
        optimizer = OptimizerSelector().select(args.optimizer)
    else:
        optimizer = Adam
    optimizer.learning_rate = args.learning_rate
    if args.snapshot_from != '':
        train_model, model_forward_run = ModelSelector().select(args.model).load(path=args.snapshot_from)
        print('Loaded model from ' + args.snapshot_from)
        trainer = Trainer(train_model=train_model, model_forward_run=model_forward_run,
                          loss_function=CrossEntropyLoss())
    else:
        train_model = ModelSelector().select(args.model).with_optimizer(optimizer)
        trainer = Trainer(train_model=train_model, model_forward_run=[], loss_function=CrossEntropyLoss())
        trainer.compile()

    loader = DatasetLoader(directory=args.dataset_dir)
    train_labels, train_data = loader.load(dataset_name='fmnist_train.npy')
    test_labels, test_data = loader.load(dataset_name='fmnist_test.npy')

    train_data, train_mean = remove_mean_on_train(train_data=train_data)
    test_data = remove_mean_on_test(test_data=test_data, train_mean=train_mean)

    trainer.train(num_iters=args.num_iters,
                  batch_size=args.batch_size,
                  test_batch_size=5000,
                  dataset=(train_labels, train_data, test_labels, test_data),
                  image_shape=(1, 28, 28),
                  snapshot=args.snapshot,
                  snapshot_dir=args.snapshot_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains specified model from scratch with specified parameters.')
    parser.add_argument("--snapshot_from", type=str, help="Path to snapshot to continue training from.", default='')
    parser.add_argument("--optimizer", type=str, help="Optimizer name.", default='')
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=0.002)
    parser.add_argument("--model", type=str, help="Model name.", default='convnet')
    parser.add_argument("--dataset_dir", type=str, help="Path to directory with data set.",
                        default=r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv')

    parser.add_argument("--num_iters", type=int, help="Number of training iterations.", default=1000)
    parser.add_argument("--batch_size", type=int, help="Training batch size.", default=64)
    parser.add_argument("--snapshot", type=int, help="Number of iterations after the next snapshot done.", default=100)
    parser.add_argument("--snapshot_dir", type=str, help="Path to a directory to save snapshots.",
                        default=r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv\model\convnet')

    _args = parser.parse_args()
    run(_args)
