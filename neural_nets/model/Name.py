from enum import Enum


class Name(Enum):
    # Layer names.
    LINEAR_TRAIN = 'Linear in train mode'
    RELU_TRAIN = 'ReLU in train mode'
    BATCH_NORM_1D_TRAIN = 'BatchNorm1D in train mode'
    BATCH_NORM_2D_TRAIN = 'BatchNorm2D in train mode'
    DROPOUT1D_TRAIN = 'Dropout 1d in train mode'
    DROPOUT2D_TRAIN = 'Spatial dropout in train mode'
    CONV2D_TRAIN = 'Convolutional in train mode'
    MAX_POOL_TRAIN = 'Max pool in train mode'

    LINEAR_TEST = 'Linear in test mode'
    RELU_TEST = 'ReLU in test mode'
    BATCH_NORM_1D_TEST = 'BatchNorm1D in test mode'
    BATCH_NORM_2D_TEST = 'BatchNORM2D in test mode'
    DROPOUT1D_TEST = 'Dropout 1d in test mode'
    DROPOUT2D_TEST = 'Spatial dropout in test mode'
    CONV2D_TEST = 'Convolutional in test mode'
    MAX_POOL_TEST = 'Max pool in test mode'

    WEIGHTS = 'weights in a linear layer'
    BIASES = 'biases in a linear layer'
    GAMMA = 'gamma in batch norm'
    BETA = 'beta in batch norm'

    # Parameter names.
    RUNNING_MEAN = 'running mean'
    RUNNING_VARIANCE = 'running variance'
    X_HAT = 'normalized input in a batch norm layer'
    IVAR = 'inverted variance'

    # Data names.
    INPUT = 'layer input'
    INPUT_FLAT = 'flatten input'
    OUTPUT = 'layer output'
    LABELS = 'data labels'
    PROBS = 'probabilities of each class for Cross Entropy Loss'
    LOSS = 'loss value on data'
    MARGINS = 'margins between scores and correct scores in SVM Loss'
    X_COLS = 'image reshaped to columns of patches'
    X_COLS_ARGMAX = 'indexes of maximum values of columns of patches'
    MASK = 'matrix of boolean values'

    # Gradients of a linear layer.
    D_WEIGHTS = 'linear weights gradient'
    D_BIASES = 'linear biases gradient'

    # Gradients of a Batch Norm Layer.
    D_GAMMA = 'gamma gradient'
    D_BETA = 'beta gradient'

    # Velocity.
    V_WEIGHTS = 'weights velocity'
    V_WEIGHTS_PREV = 'previous weights velocity'

    V_KERNEL_WEIGHTS = 'kernel weights velocity'
    V_KERNEL_WEIGHTS_PREV = 'previous kernel weights velocity'

    V_KERNEL_BIASES = 'kernel biases velocity'
    V_KERNEL_BIASES_PREV = 'previous kernel biases velocity'

    V_BIASES = 'biases velocity'
    V_BIASES_PREV = 'previous biases velocity'

    V_GAMMA = 'gamma velocity'
    V_GAMMA_PREV = 'previous gamma velocity'

    V_BETA = 'beta velocity'
    V_BETA_PREV = 'previous beta velocity'

    # Cache names for adaptive learning rate methods.
    D_WEIGHTS_CACHE = 'cache for weights gradients'
    D_BIASES_CACHE = 'cache for biases gradients'

    D_GAMMA_CACHE = 'cache for gamma gradients'
    D_BETA_CACHE = 'cache for beta gradients'
