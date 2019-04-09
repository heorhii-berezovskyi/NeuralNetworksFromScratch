import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayerWithWeights, TestModeLayer
from neural_nets.model.Name import Name
from neural_nets.utils.DatasetProcessingUtils import im2col_indices, col2im_indices


class Conv2DTest(TestModeLayer):
    def __init__(self, weights: Cache, stride: int, padding: int):
        super().__init__()
        self.weights = weights
        self.num_filters = weights.get(name=Name.WEIGHTS).shape[0]
        self.filter_height = weights.get(name=Name.WEIGHTS).shape[2]
        self.filter_width = weights.get(name=Name.WEIGHTS).shape[3]
        self.stride = stride
        self.padding = padding

    def get_id(self) -> int:
        return self.id

    def get_name(self) -> Name:
        return Name.CONV2D_TEST

    def forward(self, input_data: ndarray) -> ndarray:
        """
          A fast implementation of the forward pass for a convolutional layer
          based on im2col and col2im.
          """
        N, C, H, W = input_data.shape

        # Check dimensions
        assert (W + 2 * self.padding - self.filter_width) % self.stride == 0, 'width does not work'
        assert (H + 2 * self.padding - self.filter_height) % self.stride == 0, 'height does not work'

        # Create output
        out_h = (H + 2 * self.padding - self.filter_height) // self.stride + 1
        out_w = (W + 2 * self.padding - self.filter_width) // self.stride + 1

        x_cols = im2col_indices(x=input_data,
                                field_height=self.filter_height,
                                field_width=self.filter_width,
                                padding=self.padding,
                                stride=self.stride)

        weights, biases = self.weights.get(name=Name.WEIGHTS), self.weights.get(name=Name.BIASES)
        res = weights.reshape((self.num_filters, -1)) @ x_cols + biases.reshape(-1, 1)

        output_data = res.reshape(self.num_filters, out_h, out_w, N)
        output_data = output_data.transpose(3, 0, 1, 2)
        return output_data

    def get_weights(self) -> Cache:
        return self.weights


class Conv2DTrain(TrainModeLayerWithWeights):
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

    def get_id(self) -> int:
        return self.id

    def get_name(self) -> Name:
        return Name.CONV2D_TRAIN

    def get_weights(self) -> Cache:
        return self.weights

    def forward(self, input_data: ndarray, test_model_params: dict) -> Cache:
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
        out_height = (H + 2 * self.padding - self.filter_height) // self.stride + 1
        out_width = (W + 2 * self.padding - self.filter_width) // self.stride + 1

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

    def backward(self, dout: ndarray, layer_forward_run: Cache) -> tuple:
        """
          A fast implementation of the backward pass for a convolutional layer
          based on im2col and col2im.
        """
        weights = self.weights.get(name=Name.WEIGHTS)
        input_data, x_cols = layer_forward_run.get(name=Name.INPUT), layer_forward_run.get(name=Name.X_COLS)

        dbiases = np.sum(dout, axis=(0, 2, 3))

        weights_shape = weights.shape
        dout_reshaped = dout.transpose((1, 2, 3, 0)).reshape(self.num_filters, -1)
        dweights = dout_reshaped.dot(x_cols.T).reshape(weights_shape)

        dx_cols = weights.reshape(self.num_filters, -1).T @ dout_reshaped
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

    def to_test(self, test_model_params: dict) -> TestModeLayer:
        layer = Conv2DTest(weights=self.weights,
                           padding=self.padding,
                           stride=self.stride)
        return layer

    def accept(self, visitor):
        visitor.visit_linear(self)

    @staticmethod
    def create_weights(num_filters: int, filter_depth: int, filter_height: int, filter_width: int) -> Cache:
        weights = Cache()
        weights.add(name=Name.WEIGHTS,
                    value=np.random.rand(num_filters, filter_depth, filter_height, filter_width) * np.sqrt(
                        2. / (filter_depth * filter_height * filter_width)))

        weights.add(name=Name.BIASES, value=np.zeros(num_filters, dtype=np.float64))
        return weights
