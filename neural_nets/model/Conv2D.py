import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TestModeLayer
from neural_nets.model.Name import Name
from neural_nets.utils.DatasetProcessingUtils import im2col_indices, col2im_indices


class Conv2DTest(TestModeLayer):
    def __init__(self, num_filters: int, filter_depth: int, filter_height: int, filter_width: int, stride: int,
                 padding: int):
        super().__init__()
        self.num_filters = num_filters
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride = stride
        self.padding = padding
        self.weights = self.create_weights(num_filters=num_filters,
                                           filter_depth=filter_depth,
                                           filter_height=filter_height,
                                           filter_width=filter_width)

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.CONV2D_TEST

    def forward(self, input_data: ndarray):
        """
          A fast implementation of the forward pass for a convolutional layer
          based on im2col and col2im.
          """
        N, C, H, W = input_data.shape

        # Check dimensions
        assert (W + 2 * self.padding - self.filter_width) % self.stride == 0, 'width does not work'
        assert (H + 2 * self.padding - self.filter_height) % self.stride == 0, 'height does not work'

        # Create output
        out_height = int((H + 2 * self.padding - self.filter_height) / self.stride + 1)
        out_width = int((W + 2 * self.padding - self.filter_width) / self.stride + 1)
        # output_data = np.zeros((N, num_filters, out_height, out_width), dtype=input_data.dtype)

        x_cols = im2col_indices(x=input_data,
                                field_height=self.filter_height,
                                field_width=self.filter_width,
                                padding=self.padding,
                                stride=self.stride)

        weights, biases = self.weights.get(name=Name.WEIGHTS), self.weights.get(name=Name.BIASES)
        res = weights.reshape((self.num_filters, -1)) @ x_cols + biases.reshape(-1, 1)

        output_data = res.reshape(self.num_filters, out_height, out_width, N)
        output_data = output_data.transpose(3, 0, 1, 2)
        return output_data

    def get_weights(self):
        return self.weights

    @staticmethod
    def create_weights(num_filters: int, filter_depth: int, filter_height: int, filter_width: int):
        weights = Cache()
        weights.add(name=Name.WEIGHTS,
                    value=0.01 * np.random.rand(num_filters, filter_depth, filter_height, filter_width))

        weights.add(name=Name.BIASES, value=np.zeros(num_filters))
        return weights


class Conv2DTrain(TrainModeLayer):
    def __init__(self, num_filters: int, filter_depth: int, filter_height: int, filter_width: int, stride: int,
                 padding: int):
        super().__init__()
        self.num_filters = num_filters
        self.filter_depth = filter_depth
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride = stride
        self.padding = padding
        self.weights = self.create_weights(num_filters=num_filters,
                                           filter_depth=filter_depth,
                                           filter_height=filter_height,
                                           filter_width=filter_width)

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.CONV2D_TRAIN

    def get_weights(self):
        return self.weights

    def forward(self, input_data: ndarray, test_model_params: dict):
        """
          A fast implementation of the forward pass for a convolutional layer
          based on im2col and col2im.
        """
        layer_forward_run = Cache()
        layer_forward_run.add(name=Name.INPUT, value=input_data)

        N, C, H, W = input_data.shape

        # Check dimensions
        assert (W + 2 * self.padding - self.filter_width) % self.stride == 0, 'width does not work'
        assert (H + 2 * self.padding - self.filter_height) % self.stride == 0, 'height does not work'

        # Create output
        out_height = int((H + 2 * self.padding - self.filter_height) / self.stride + 1)
        out_width = int((W + 2 * self.padding - self.filter_width) / self.stride + 1)
        # output_data = np.zeros((N, num_filters, out_height, out_width), dtype=input_data.dtype)

        x_cols = im2col_indices(x=input_data,
                                field_height=self.filter_height,
                                field_width=self.filter_width,
                                padding=self.padding,
                                stride=self.stride)

        weights, biases = self.weights.get(name=Name.WEIGHTS), self.weights.get(name=Name.BIASES)
        res = weights.reshape((self.num_filters, -1)) @ x_cols + biases.reshape(-1, 1)

        output_data = res.reshape(self.num_filters, out_height, out_width, N)
        output_data = output_data.transpose(3, 0, 1, 2)

        layer_forward_run.add(name=Name.X_COLS, value=x_cols)
        layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache):
        """
          A fast implementation of the backward pass for a convolutional layer
          based on im2col and col2im.
        """
        input_data, x_cols = layer_forward_run.get(name=Name.INPUT), layer_forward_run.get(name=Name.X_COLS)

        dbiases = np.sum(dout, axis=(0, 2, 3))

        weights_shape = self.weights.get(name=Name.WEIGHTS).shape
        dout_reshaped = dout.transpose((1, 2, 3, 0)).reshape(self.num_filters, -1)
        dweights = dout_reshaped.dot(x_cols.T).reshape(weights_shape)

        dx_cols = self.weights.get(name=Name.WEIGHTS).reshape(self.num_filters, -1).T.dot(dout_reshaped)
        dinput = col2im_indices(cols=dx_cols,
                                x_shape=input_data.shape,
                                field_height=self.filter_height,
                                field_width=self.filter_width,
                                padding=self.padding,
                                stride=self.stride)

        layer_backward_run = Cache()
        layer_backward_run.add(name=Name.D_BIASES, value=dbiases)
        layer_backward_run.add(name=Name.D_WEIGHTS, value=dweights)
        return dinput, layer_backward_run

    def to_test(self, test_model_params: dict):
        weights = self.weights
        layer = Conv2DTest(num_filters=self.num_filters,
                           filter_depth=self.filter_depth,
                           filter_height=self.filter_height,
                           filter_width=self.filter_width,
                           padding=self.padding,
                           stride=self.stride)
        layer.get_weights().update(name=Name.WEIGHTS, value=weights.get(name=Name.WEIGHTS))
        layer.get_weights().update(name=Name.BIASES, value=weights.get(name=Name.BIASES))
        return layer

    def accept(self, visitor):
        visitor.visit_conv2d(self)

    @staticmethod
    def create_weights(num_filters: int, filter_depth: int, filter_height: int, filter_width: int):
        weights = Cache()
        weights.add(name=Name.WEIGHTS,
                    value=0.01 * np.random.rand(num_filters, filter_depth, filter_height, filter_width))

        weights.add(name=Name.BIASES, value=np.zeros(num_filters))
        return weights
