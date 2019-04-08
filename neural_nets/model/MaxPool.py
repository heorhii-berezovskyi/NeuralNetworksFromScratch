import numpy as np
from numpy import ndarray

from neural_nets.model.Cache import Cache
from neural_nets.model.Layer import TrainModeLayer, TestModeLayer
from neural_nets.model.Name import Name
from neural_nets.utils.DatasetProcessingUtils import im2col_indices, col2im_indices


class MaxPoolTest(TestModeLayer):
    def __init__(self, pool_height: int, pool_width: int, stride: int):
        super().__init__()
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.MAX_POOL_TEST

    def forward(self, input_data: ndarray):
        N, C, H, W = input_data.shape

        assert (H - self.pool_height) % self.stride == 0, 'Invalid height'
        assert (W - self.pool_width) % self.stride == 0, 'Invalid width'

        out_h = (H - self.pool_height) // self.stride + 1
        out_w = (W - self.pool_width) // self.stride + 1

        x_split = input_data.reshape((N * C, 1, H, W))
        x_cols = im2col_indices(x_split, self.pool_height, self.pool_width, padding=0, stride=self.stride)
        x_cols_argmax = np.argmax(x_cols, axis=0)
        x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
        out = x_cols_max.reshape(out_h, out_w, N, C).transpose(2, 3, 0, 1)
        return out


class MaxPoolTrain(TrainModeLayer):
    def __init__(self, pool_height: int, pool_width: int, stride: int):
        super().__init__()
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride

    def get_id(self):
        return self.id

    def get_name(self):
        return Name.MAX_POOL_TRAIN

    def forward(self, input_data: ndarray, test_model_params: dict):
        N, C, H, W = input_data.shape

        assert (H - self.pool_height) % self.stride == 0, 'Invalid height'
        assert (W - self.pool_width) % self.stride == 0, 'Invalid width'

        out_h = 1 + (H - self.pool_height) // self.stride
        out_w = 1 + (W - self.pool_width) // self.stride

        x_split = input_data.reshape((N * C, 1, H, W))
        x_cols = im2col_indices(x_split, self.pool_height, self.pool_width, padding=0, stride=self.stride)
        x_cols_argmax = np.argmax(x_cols, axis=0)
        x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
        output_data = x_cols_max.reshape(out_h, out_w, N, C).transpose(2, 3, 0, 1)

        layer_forward_run = Cache()
        layer_forward_run.add(name=Name.INPUT, value=input_data)
        layer_forward_run.add(name=Name.OUTPUT, value=output_data)
        layer_forward_run.add(name=Name.X_COLS, value=x_cols)
        layer_forward_run.add(name=Name.X_COLS_ARGMAX, value=x_cols_argmax)
        return layer_forward_run

    def backward(self, dout: ndarray, layer_forward_run: Cache):
        input_data = layer_forward_run.get(name=Name.INPUT)
        x_cols = layer_forward_run.get(name=Name.X_COLS)
        x_cols_argmax = layer_forward_run.get(name=Name.X_COLS_ARGMAX)

        N, C, H, W = input_data.shape

        dout_reshaped = dout.transpose((2, 3, 0, 1)).flatten()
        dx_cols = np.zeros_like(x_cols)
        dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
        dinput = col2im_indices(dx_cols, (N * C, 1, H, W), self.pool_height, self.pool_width,
                                padding=0, stride=self.stride)
        dinput = dinput.reshape(input_data.shape)

        layer_backward_run = Cache()
        return dinput, layer_backward_run

    def to_test(self, test_model_params: dict):
        layer = MaxPoolTest(pool_height=self.pool_height,
                            pool_width=self.pool_width,
                            stride=self.stride)
        return layer

    def accept(self, visitor):
        visitor.visit_max_pool(self)
