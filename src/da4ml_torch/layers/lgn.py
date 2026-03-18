import numpy as np
from da4ml.trace import FixedVariableArray

from ._base import ReplayBase
from ._standalone_inference import GroupSum, OrPooling2d
from torchlogix.layers import LogicConv2d, LogicDense

_map = [
    lambda a, b: 0,
    lambda a, b: a & b,
    lambda a, b: a & ~b,
    lambda a, b: a,
    lambda a, b: b & ~a,
    lambda a, b: b,
    lambda a, b: a ^ b,
    lambda a, b: a | b,
    lambda a, b: ~(a | b),
    lambda a, b: ~(a ^ b),
    lambda a, b: ~b,
    lambda a, b: ~(b & ~a),
    lambda a, b: ~a,
    lambda a, b: ~(a & ~b),
    lambda a, b: ~(a & b),
    lambda a, b: 1,
]


def apply_lut_vectorized(a, b, lut_ids):
    solver_options = a.solver_options
    a, b = np.array(a), np.array(b)
    result = np.empty_like(a)
    for lut_id in range(16):
        mask = lut_ids == lut_id
        result[..., mask] = _map[lut_id](a[..., mask], b[..., mask])
    return FixedVariableArray(result, solver_options=solver_options)


class ReplayLogicConv2d(ReplayBase):
    handles = (LogicConv2d,)

    def call(self, inputs: FixedVariableArray):
        self.module: LogicConv2d

        # Safety check: model must be in eval mode for discrete LUT extraction
        assert not self.module.training, (
            "LogicConv2d module must be in eval mode for synthesis. "
            "Call model.eval() before tracing."
        )

        # Apply padding if needed
        if self.module.padding > 0:
            inputs = np.pad(
                inputs,  # type: ignore
                ((0, 0), (self.module.padding, self.module.padding), (self.module.padding, self.module.padding)),
                mode='constant',
                constant_values=0,
            )  # type: ignore

        # Get all LUTs and connection indices for the entire tree
        tree_luts, tree_ids = self.module.get_luts_and_ids()
        conn_indices = self.module.connections.indices

        # Level 0: Extract from receptive field
        # conn_indices[0] shape: [lut_rank, num_kernels, num_positions, num_leaves, 3]
        conn_0: np.ndarray = conn_indices[0].cpu().detach().numpy()  # type: ignore

        # tree_ids[0] is a list of tensors, one per node at level 0
        # Each tree_ids[0][i] has shape [num_kernels]
        # Stack them to get shape [num_leaves, num_kernels]
        lut_0_list = [tree_ids[0][i].cpu().detach().numpy() for i in range(len(tree_ids[0]))]
        lut_0 = np.stack(lut_0_list, axis=0)  # Shape: [num_leaves, num_kernels]

        # Extract h, w, c indices
        h_idx = conn_0[..., 0]  # shape: [2, num_kernels, num_positions, num_leaves]
        w_idx = conn_0[..., 1]
        c_idx = conn_0[..., 2]

        # Select from input using advanced indexing
        # x shape: [batch, channels, H, W]
        # We want: [batch, 2, num_kernels, num_positions, num_leaves]
        selected = inputs[:, c_idx, h_idx, w_idx]  # [batch, 2, K, P, L]

        # Apply LUTs at level 0
        # lut_0 shape: [num_leaves, num_kernels]
        a = selected[:, 0]  # [batch, K, P, L]
        b = selected[:, 1]

        # Reshape for vectorized application
        batch, K, P, L = a.shape
        a_flat = a.transpose((0, 2, 1, 3)).reshape(batch * P, K * L)  # type: ignore
        b_flat = b.transpose((0, 2, 1, 3)).reshape(batch * P, K * L)  # type: ignore
        lut_0_flat = lut_0.T.flatten()  # [K*L]

        result = apply_lut_vectorized(a_flat, b_flat, lut_0_flat)
        result = result.reshape(batch, P, K, L).transpose((0, 2, 1, 3))  # [batch, K, P, L]

        # Process remaining tree levels
        for level in range(1, self.module.tree_depth):
            # tree_ids[level] is a list of tensors for this level
            # Stack them to get shape [num_nodes, num_kernels]
            lut_level_list = [tree_ids[level][i].cpu().detach().numpy() for i in range(len(tree_ids[level]))]
            lut_level = np.stack(lut_level_list, axis=0)  # Shape: [num_nodes, num_kernels]

            # conn_indices[level] for level > 0 has shape [lut_rank, num_nodes]
            conn_level: np.ndarray = conn_indices[level].cpu().detach().numpy()  # type: ignore

            selected = result[..., conn_level]

            a = selected[..., 0, :]
            b = selected[..., 1, :]

            batch, K, P, N = a.shape
            a_flat = a.transpose((0, 2, 1, 3)).reshape(batch * P, K * N)
            b_flat = b.transpose((0, 2, 1, 3)).reshape(batch * P, K * N)
            lut_flat = lut_level.T.flatten()  # [K*N]

            result = apply_lut_vectorized(a_flat, b_flat, lut_flat)
            result = result.reshape(batch, P, K, N).transpose((0, 2, 1, 3))  # [batch, K, P, N]

        # Calculate output dimensions dynamically
        out_dim = tuple([
            (in_d + 2 * self.module.padding - rf) // self.module.stride + 1
            for in_d, rf in zip(self.module.in_dim, self.module.receptive_field_size)
        ])

        result = result[..., 0]
        result = result.reshape(batch, self.module.num_kernels, *out_dim)

        return result


class ReplayLogicDense(ReplayBase):
    handles = (LogicDense,)

    def call(self, inputs: FixedVariableArray):
        self.module: LogicDense

        # Safety check: model must be in eval mode for discrete LUT extraction
        assert not self.module.training, (
            "LogicDense module must be in eval mode for synthesis. "
            "Call model.eval() before tracing."
        )

        # Extract connection indices from connections object
        connection_indices = self.module.connections.indices.cpu().detach().numpy()  # type: ignore
        selected = inputs[..., connection_indices]

        # Extract first and second inputs for each neuron
        a = selected[..., 0, :]  # shape: (..., out_dim)
        b = selected[..., 1, :]  # shape: (..., out_dim)

        # Get discrete LUTs and IDs using parametrization method
        luts, lut_ids = self.module.get_luts_and_ids()
        lut_ids = lut_ids.cpu().detach().numpy()  # type: ignore

        # Apply logic operations
        return apply_lut_vectorized(a, b, lut_ids)


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
