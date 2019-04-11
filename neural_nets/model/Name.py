from enum import Enum


class Name(Enum):
    # Layer names.
    LINEAR_TRAIN = 'linear_train'
    RELU_TRAIN = 'relu_train'
    BATCH_NORM_1D_TRAIN = 'batchnorm1d_train'
    BATCH_NORM_2D_TRAIN = 'batchnorm2d_train'
    DROPOUT1D_TRAIN = 'dropout1d_train'
    DROPOUT2D_TRAIN = 'dropout2d_train'
    CONV2D_TRAIN = 'conv2d_train'
    MAX_POOL_TRAIN = 'maxpool_train'

    LINEAR_TEST = 'linear_test'
    RELU_TEST = 'relu_test'
    BATCH_NORM_1D_TEST = 'batchnorm1d_test'
    BATCH_NORM_2D_TEST = 'batchnorm2d_test'
    DROPOUT1D_TEST = 'dropout1d_test'
    DROPOUT2D_TEST = 'dropout2d_test'
    CONV2D_TEST = 'conv2d_test'
    MAX_POOL_TEST = 'maxpool_test'

    WEIGHTS = 'weights'
    BIASES = 'biases'
    GAMMA = 'gamma'
    BETA = 'beta'

    # Parameter names.
    RUNNING_MEAN = 'running_mean'
    RUNNING_VARIANCE = 'running_variance'
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

    ADAM_M_WEIGHTS = 'm cache for weights'
    ADAM_V_WEIGHTS = 'v cache for weights'

    ADAM_M_BIASES = 'm cache for biases'
    ADAM_V_BIASES = 'v cache for biases'

    ADAM_M_GAMMA = 'm cache for gamma'
    ADAM_V_GAMMA = 'v cache for gamma'

    ADAM_M_BETA = 'm cache for beta'
    ADAM_V_BETA = 'v cache for beta'
