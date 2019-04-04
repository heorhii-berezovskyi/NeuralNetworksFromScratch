from enum import Enum


class Name(Enum):
    # Layer names.
    LINEAR_TRAIN = 'Linear in train mode'
    RELU_TRAIN = 'ReLU in train mode'
    BATCH_NORM_1D_TRAIN = 'BatchNorm1D in train mode'
    BATCH_NORM_2D_TRAIN = 'BatchNorm2D in train mode'
    DROPOUT1D_TRAIN = 'Dropout in train mode'
    CONV2D_TRAIN = 'Convolutional in train mode'
    MAX_POOL_TRAIN = 'Max pool in train mode'

    LINEAR_TEST = 'Linear in test mode'
    RELU_TEST = 'ReLU in test mode'
    BATCH_NORM_1D_TEST = 'BatchNorm1D in test mode'
    BATCH_NORM_2D_TEST = 'BatchNORM2D in test mode'
    DROPOUT1D_TEST = 'Dropout in test mode'
    CONV2D_TEST = 'Convolutional in test mode'
    MAX_POOL_TEST = 'Max pool in test mode'

    # Weights names.
    WEIGHTS = 'weights'
    BIASES = 'biases'
    GAMMA = 'gamma'
    BETA = 'beta'

    # Parameter names.
    RUNNING_MEAN = 'running mean'
    RUNNING_VARIANCE = 'running variance'
    MU = 'mini-batch mean'
    VAR = 'mini-batch variance'

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
    D_WEIGHTS = 'weights gradient'
    D_BIASES = 'biases gradient'

    # Gradients of a Batch Norm Layer.
    D_GAMMA = 'gamma gradient'
    D_BETA = 'beta gradient'

    # Velocity.
    V_WEIGHTS = 'weights velocity'
    V_WEIGHTS_PREV = 'previous weights velocity'

    V_BIASES = 'biases velocity'
    V_BIASES_PREV = 'previous biases velocity'

    V_GAMMA = 'gamma velocity'
    V_GAMMA_PREV = 'previous gamma velocity'

    V_BETA = 'beta velocity'
    V_BETA_PREV = 'previous beta velocity'
