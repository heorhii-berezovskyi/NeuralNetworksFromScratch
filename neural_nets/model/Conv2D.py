import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayerWithWeights, TestModeLayerWithWeights
from neural_nets.model.Name import Name
from neural_nets.model.Visitor import TrainLayerVisitor, TestLayerVisitor
from neural_nets.optimizer.Optimizer import Optimizer
from neural_nets.utils.DatasetProcessingUtils import im2col_indices, col2im_indices


class Conv2DTest(TestModeLayerWithWeights):
    name = Name.CONV2D_TEST

    def __init__(self, layer_id: int, weights: Cache, stride: int, padding: int):
        self.id = layer_id
        self.weights = weights
        self.num_filters = weights.get(name=Name.WEIGHTS).shape[0]
        self.filter_height = weights.get(name=Name.WEIGHTS).shape[2]
        self.filter_width = weights.get(name=Name.WEIGHTS).shape[3]
        self.stride = stride
        self.padding = padding

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

    def content(self) -> dict:
        layer_id = Conv2DTest.name.value + str(self.id)
        result = {}
        for item_name in self.weights.get_keys():
            data_value = self.weights.get(name=item_name)
            data_key = layer_id + item_name.value
            result[data_key] = data_value
        return result

    def from_params(self, all_params):
        weights = Cache()

        layer_id = Conv2DTest.name.value + str(self.id)
        for w_name in [Name.WEIGHTS, Name.BIASES]:
            w_key = layer_id + w_name.value
            w_value = all_params[w_key]
            weights.add(name=w_name, value=w_value)
        return Conv2DTest(layer_id=self.id, weights=weights, stride=self.stride, padding=self.padding)

    def accept(self, visitor: TestLayerVisitor):
        visitor.visit_weighted_test(self)


class Conv2DTrain(TrainModeLayerWithWeights):
    name = Name.CONV2D_TRAIN

    def __init__(self, layer_id: int, weights: Cache, stride: int, padding: int, optimizer: Optimizer):
        self.id = layer_id
        self.optimizer = optimizer
        self.stride = stride
        self.padding = padding
        self.weights = weights
        self.num_filters = weights.get(name=Name.WEIGHTS).shape[0]
        self.filter_height = weights.get(name=Name.WEIGHTS).shape[2]
        self.filter_width = weights.get(name=Name.WEIGHTS).shape[3]

    def forward(self, input_data: ndarray, layer_forward_run: Cache) -> Cache:
        """
          A fast implementation of the forward pass for a convolutional layer
          based on im2col and col2im.
        """
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

        new_layer_forward_run = Cache()
        new_layer_forward_run.add(name=Name.INPUT, value=input_data)
        new_layer_forward_run.add(name=Name.X_COLS, value=x_cols)
        new_layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        return new_layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache) -> Cache:
        """
          A fast implementation of the backward pass for a convolutional layer
          based on im2col and col2im.
        """
        weights = self.weights.get(name=Name.WEIGHTS)
        input_data, x_cols = layer_forward_run.pop(name=Name.INPUT), layer_forward_run.pop(name=Name.X_COLS)

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
        layer_backward_run.add(name=Name.D_INPUT, value=dinput)
        layer_backward_run.add(name=Name.WEIGHTS, value=dweights)
        layer_backward_run.add(name=Name.BIASES, value=dbiases)
        return layer_backward_run

    def to_test(self, test_layer_params: Cache) -> TestModeLayerWithWeights:
        return Conv2DTest(layer_id=self.id,
                          weights=self.weights,
                          padding=self.padding,
                          stride=self.stride)

    def optimize(self, layer_backward_run: Cache) -> TrainModeLayerWithWeights:
        new_optimizer = self.optimizer.update_memory(layer_backward_run=layer_backward_run)
        new_weights = new_optimizer.update_weights(self.weights)
        return Conv2DTrain(layer_id=self.id,
                           weights=new_weights,
                           stride=self.stride,
                           padding=self.padding,
                           optimizer=new_optimizer)

    def accept(self, visitor: TrainLayerVisitor):
        visitor.visit_affine_train(self)

    @staticmethod
    def init_weights(num_filters: int, filter_depth: int, filter_height: int, filter_width: int) -> Cache:
        weights = Cache()
        weights.add(name=Name.WEIGHTS,
                    value=np.random.rand(num_filters, filter_depth, filter_height, filter_width) * np.sqrt(
                        2. / (filter_depth * filter_height * filter_width)))
        weights.add(name=Name.BIASES, value=np.zeros(num_filters, dtype=float))
        return weights
