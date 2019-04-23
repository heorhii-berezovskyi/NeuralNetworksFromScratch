from neural_nets.dataset.DatasetLoader import DatasetLoader
from neural_nets.model.CrossEntropyLoss import CrossEntropyLoss
from neural_nets.model.Trainer import Trainer
from neural_nets.optimizer.Adam import Adam
from neural_nets.utils.DatasetProcessingUtils import remove_mean_on_train, remove_mean_on_test

if __name__ == "__main__":
    # SGDMomentum.learning_rate = 0.0001

    # SGDNesterovMomentum.learning_rate = 0.02
    # SGDNesterovMomentum.mu = 0.9

    # RMSprop.learning_rate = 0.02
    # RMSprop.decay_rate = 0.9

    Adam.learning_rate = 0.005
    Adam.beta1 = 0.9
    Adam.beta2 = 0.999

    # Adagrad.learning_rate = 0.02

    trainer = Trainer(optimizer=Adam, loss_function=CrossEntropyLoss())

    ######################################

    trainer.add_conv2d(block_name='conv1',
                       num_filters=32,
                       filter_depth=1,
                       filter_height=3,
                       filter_width=3,
                       stride=1,
                       padding=1)
    trainer.add_batch_norm2d(block_name='conv1',
                             num_of_channels=32,
                             momentum=0.9)
    trainer.add_relu()
    trainer.add_dropout2d(keep_active_prob=0.9)
    trainer.add_max_pool(pool_height=2, pool_width=2, stride=2)

    ######################################

    trainer.add_conv2d(block_name='conv2',
                       num_filters=64,
                       filter_depth=32,
                       filter_height=3,
                       filter_width=3,
                       stride=1,
                       padding=1)
    trainer.add_batch_norm2d(block_name='conv2',
                             num_of_channels=64,
                             momentum=0.9)
    trainer.add_relu()
    trainer.add_dropout2d(keep_active_prob=0.7)
    trainer.add_max_pool(pool_height=2, pool_width=2, stride=2)

    ######################################

    trainer.add_linear(block_name='linear1',
                       input_dim=3136,
                       num_of_neurons=128)
    trainer.add_batch_norm1d(block_name='linear1',
                             input_dim=128,
                             momentum=0.9)
    trainer.add_relu()
    trainer.add_dropout1d(keep_active_prob=0.5)

    ######################################

    trainer.add_linear(block_name='linear2',
                       input_dim=128,
                       num_of_neurons=10)

    ######################################

    trainer.compile()
    # trainer.load_model(path=r'200.npz')

    loader = DatasetLoader()

    train_labels, train_data, test_labels, test_data = loader.load(train_dataset_name='fmnist_train.npy',
                                                                   test_dataset_name='fmnist_test.npy')

    train_data, train_mean = remove_mean_on_train(train_data=train_data)
    test_data = remove_mean_on_test(test_data=test_data, train_mean=train_mean)

    dataset = (train_labels, train_data, test_labels, test_data)

    trainer.train(num_epoch=1000,
                  batch_size=64,
                  test_batch_size=5000,
                  dataset=dataset,
                  image_shape=(1, 28, 28),
                  snapshot=200)

    # image = train_data[3]
    # image = image.reshape(1, 28, 28)
    # print(trainer.predict(image=image))
