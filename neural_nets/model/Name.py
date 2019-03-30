from enum import Enum


class Name(Enum):
    # Layer names.
    LINEAR_TRAIN = 'Linear in train mode'
    RELU_TRAIN = 'ReLU in train mode'
    BATCH_NORM_TRAIN = 'BatchNorm in train mode'
    CONV2D_TRAIN = 'Convolutional in train mode'

    LINEAR_TEST = 'Linear in test mode'
    RELU_TEST = 'ReLU in test mode'
    BATCH_NORM_TEST = 'BatchNorm in test mode'
    CONV2D_TEST = 'Convolutional in test mode'

    # Weights names.
    WEIGHTS = 'weights'
    BIASES = 'biases'
    GAMMA = 'gamma'
    BETA = 'beta'

    # Parameter names.
    RUNNING_MEAN = 'running mean'
    RUNNING_VARIANCE = 'running variance'
    MU = 'mean'
    VAR = 'variance'

    # Data names.
    INPUT = 'layer input'
    OUTPUT = 'layer output'
    LABELS = 'data labels'
    PROBS = 'probabilities of each class for Cross Entropy Loss'
    LOSS = 'loss value on data'
    MARGINS = 'margins between scores and correct scores in SVM Loss'
    X_COLS = 'image reshaped to columns of patches'
    RESHAPED_INPUT_DATA = 'image reshaped to matrix [N X D]'

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
