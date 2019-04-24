from neural_nets.model.BatchNorm1D import BatchNorm1DTrain
from neural_nets.model.BatchNorm2D import BatchNorm2DTrain
from neural_nets.model.Conv2D import Conv2DTrain
from neural_nets.model.Dropout1D import Dropout1DTrain
from neural_nets.model.Dropout2D import Dropout2DTrain
from neural_nets.model.Linear import LinearTrain
from neural_nets.model.MaxPool import MaxPoolTrain
from neural_nets.model.Model import TrainModel
from neural_nets.model.Relu import ReluTrain


def net() -> TrainModel:
    layers = list()
    layers.append(Conv2DTrain.init(block_name='conv1',
                                   num_filters=32,
                                   filter_depth=1,
                                   filter_height=3,
                                   filter_width=3,
                                   stride=1,
                                   padding=1))

    layers.append(BatchNorm2DTrain.init(block_name='conv1',
                                        num_of_channels=32,
                                        momentum=0.9))
    layers.append(ReluTrain())
    layers.append(Dropout2DTrain(keep_active_prob=0.9))
    layers.append(MaxPoolTrain(pool_height=2, pool_width=2, stride=2))

    ######################################

    layers.append(Conv2DTrain.init(block_name='conv2',
                                   num_filters=64,
                                   filter_depth=32,
                                   filter_height=3,
                                   filter_width=3,
                                   stride=1,
                                   padding=1))
    layers.append(BatchNorm2DTrain.init(block_name='conv2',
                                        num_of_channels=64,
                                        momentum=0.9))
    layers.append(ReluTrain())
    layers.append(Dropout2DTrain(keep_active_prob=0.7))
    layers.append(MaxPoolTrain(pool_height=2, pool_width=2, stride=2))

    ######################################

    layers.append(LinearTrain.init(block_name='linear1',
                                   input_dim=3136,
                                   num_of_neurons=128))
    layers.append(BatchNorm1DTrain.init(block_name='linear1',
                                        input_dim=128,
                                        momentum=0.9))
    layers.append(ReluTrain())
    layers.append(Dropout1DTrain(keep_active_prob=0.5))

    ######################################

    layers.append(LinearTrain.init(block_name='linear2',
                                   input_dim=128,
                                   num_of_neurons=10))
    return TrainModel(layers=layers)
