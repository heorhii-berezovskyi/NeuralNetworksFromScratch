from enum import Enum


class Name(Enum):
    # Layer names.
    LINEAR = 'linear_train'
    RELU = 'relu_train'
    BATCH_NORM_1D = 'batchnorm1d_train'
    BATCH_NORM_2D = 'batchnorm2d_train'
    DROPOUT1D = 'dropout1d_train'
    DROPOUT2D = 'dropout2d_train'
    CONV2D = 'conv2d_train'
    MAX_POOL = 'maxpool_train'

    WEIGHTS = 'weights'
    BIASES = 'biases'
    GAMMA = 'gamma'
    BETA = 'beta'

    # Parameter names.
    RUNNING_MEAN = 'running_mean'
    RUNNING_VAR = 'running_variance'
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

    D_INPUT = 'gradient of a layer by its input'

    # Velocity.
    VEL = 'velocity'
    VEL_PREV = 'previous_velocity'

    FIRST_MOM = 'first_moment'
    SECOND_MOM = 'second_moment'

    NUM_ITER = 'iteration_number'

    # Optimizer name.
    ADAGRAD = 'Adagrad'
    ADAM = 'Adam'
    RMSPROP = 'RMSprop'
    SGD_MOMENTUM = 'SGDMomentum'
    SGD_NESTEROV_MOMENTUM = 'SGDNesterovMomentum'

    LEARNING_RATE = 'learning_rate'
    BETA_1 = 'beta1'
    BETA_2 = 'beta2'

    DECAY_RATE = 'decay_rate'

    MU = 'mu'
