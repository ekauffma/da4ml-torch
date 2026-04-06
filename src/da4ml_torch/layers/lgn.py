import numpy as np
from da4ml.trace import FixedVariableArray

from ._base import ReplayBase
from torchlogix.layers import LogicConv2d, LogicDense, GroupSum, OrPooling2d

class ReplayLogicConv2d(ReplayBase):
    handles = (LogicConv2d,)

    def call(self, inputs: FixedVariableArray):
        self.module: LogicConv2d

        # Save solver_options for conversion back
        solver_options = inputs.solver_options

        # Convert FixedVariableArray to numpy array of FixedVariable objects
        inputs_np = np.array(inputs)

        # Call torchlogix forward (which now works with numpy arrays)
        result_np = self.module(inputs_np)

        # Convert back to FixedVariableArray
        return FixedVariableArray(result_np, solver_options=solver_options)


class ReplayLogicDense(ReplayBase):
    handles = (LogicDense,)

    def call(self, inputs: FixedVariableArray):
        self.module: LogicDense

        # Save solver_options for conversion back
        solver_options = inputs.solver_options

        # Convert FixedVariableArray to numpy array of FixedVariable objects
        inputs_np = np.array(inputs)

        # Call torchlogix forward (which now works with numpy arrays)
        result_np = self.module(inputs_np)

        # Convert back to FixedVariableArray
        return FixedVariableArray(result_np, solver_options=solver_options)


def im2col(inp, px_in):
    inp_col = np.lib.stride_tricks.sliding_window_view(  # type: ignore
        inp,  # type: ignore
        px_in,
        axis=tuple(range(len(px_in))),  # type: ignore
    )
    inp_col = np.moveaxis(inp_col, len(px_in), -1).reshape(*inp_col.shape[: len(px_in)], -1)
    return inp_col


class ReplayOrPooling2d(ReplayBase):
    handles = (OrPooling2d,)

    def call(self, inputs: FixedVariableArray):
        self.module: OrPooling2d

        assert inputs.ndim == 4, 'Input tensor must be 4d'
        ker_size = self.module.kernel_size
        if not isinstance(ker_size, tuple):
            ker_size = (ker_size, ker_size)
        stride = self.module.stride
        if stride is None:
            stride = ker_size
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        padding = self.module.padding

        if padding > 0:
            inputs = np.pad(
                inputs,  # type: ignore
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode='constant',
                constant_values=0,
            )  # type: ignore

        ch = inputs.shape[1]
        inp = np.moveaxis(inputs, 1, -1)  # type: ignore
        inp = im2col(inp[0], ker_size)
        inp = inp.reshape(inp.shape[:-1] + (-1, ch))[None]
        out = np.any(inp, axis=-2)
        out: FixedVariableArray = np.moveaxis(out, -1, 1)  # type: ignore
        out = out[:, :, :: stride[0], :: stride[1]]
        return out


class ReplayGroupSum(ReplayBase):
    handles = (GroupSum,)

    def call(self, inputs: FixedVariableArray):
        x = inputs
        x = x.reshape(*x.shape[:-1], self.module.k, x.shape[-1] // self.module.k)
        return (np.sum(x, -1) + self.module.beta) / self.module.tau  # type: ignore
