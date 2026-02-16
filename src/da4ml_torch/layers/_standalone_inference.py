"""
Self-contained inference code for Logic Gate Networks.
This file contains simplified, eval-mode-only implementations of LogicConv2d,
LogicDense, and GroupSum layers for hardware synthesis purposes.
"""
# import numpy
import torch
import torch.nn as nn


# ============================================================================
# 16 Boolean Logic Operations on 2 Inputs
# ============================================================================
# Each function takes two inputs (a, b) and returns the result of a specific
# Boolean operation. Inputs are in [0, 1] representing False and True.
# Lookup table: maps LUT ID (0-15) to the corresponding logic function
LOGIC_OPS = [
    lambda a, b: torch.zeros_like(a),
    lambda a, b: a * b,
    lambda a, b: a - a * b,
    lambda a, b: a,
    lambda a, b: b - a * b,
    lambda a, b: b,
    lambda a, b: a + b - 2 * a * b,
    lambda a, b: a + b - a * b,
    lambda a, b: 1 - (a + b - a * b),
    lambda a, b: 1 - (a + b - 2 * a * b),
    lambda a, b: 1 - b,
    lambda a, b: 1 - b + a * b,
    lambda a, b: 1 - a,
    lambda a, b: 1 - a + a * b,
    lambda a, b: 1 - a * b,
    lambda a, b: torch.ones_like(a),
]

def apply_lut_vectorized(a, b, lut_ids):
    """
    Apply multiple 2-input logic gates in parallel.
    Args:
        a: First inputs (shape: [..., N])
        b: Second inputs (shape: [..., N])
        lut_ids: LUT IDs for each gate (shape: [N])
    Returns:
        Result tensor (shape: [..., N])
    """
    result = torch.zeros_like(a)
    for lut_id in range(16):
        mask = (lut_ids == lut_id)
        if mask.any():
            result[..., mask] = LOGIC_OPS[lut_id](a[..., mask], b[..., mask])
    return result


class GroupSum(nn.Module):
    """
    Aggregation layer that sums groups of neurons.
    Takes N input neurons and produces k outputs by summing every (N/k) neurons.
    Used for classification, where k is the number of classes.
    Args:
        k: Number of output groups (e.g., number of classes)
        tau: Temperature for normalization (divides the summed values)
        beta: Bias term added to sums before division
    """

    def __init__(self, k, tau=1.0, beta=0.0):
        super().__init__()
        self.k = k
        self.tau = tau
        self.beta = beta

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., N) where N % k == 0

        Returns:
            Output tensor of shape (..., k)
        """
        assert x.shape[-1] % self.k == 0, \
            f"Input dimension {x.shape[-1]} must be divisible by k={self.k}"

        # Reshape to (..., k, N/k) and sum over last dimension
        x = x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k)
        return (x.sum(-1) + self.beta) / self.tau

    def __repr__(self):
        return f"GroupSum(k={self.k}, tau={self.tau}, beta={self.beta})"


class LogicDense(nn.Module):
    """
    Fully-connected layer of 2-input logic gates.
    Each output neuron:
    1. Selects 2 specific inputs from the input vector (via connection indices)
    2. Applies a learned 2-input Boolean function (specified by LUT ID)
    Args:
        in_dim: Number of input features
        out_dim: Number of output neurons
        connection_indices: Tensor of shape [2, out_dim] specifying which 2 inputs
                           each neuron connects to
        lut_ids: Tensor of shape [out_dim] specifying which logic operation
                each neuron performs (integer in [0, 15])
    """

    def __init__(self, in_dim, out_dim, connection_indices, lut_ids):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Register as buffers (not parameters, since they're frozen)
        self.register_buffer('connection_indices', connection_indices)
        self.register_buffer('lut_ids', lut_ids)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., in_dim)
        Returns:
            Output tensor of shape (..., out_dim)
        """
        assert x.shape[-1] == self.in_dim, \
            f"Expected input dimension {self.in_dim}, got {x.shape[-1]}"

        # Select inputs: shape (..., 2, out_dim)
        selected = x[..., self.connection_indices]

        # Extract first and second inputs for each neuron
        a = selected[..., 0, :]  # shape: (..., out_dim)
        b = selected[..., 1, :]  # shape: (..., out_dim)

        # Apply logic operations
        return apply_lut_vectorized(a, b, self.lut_ids)

    def __repr__(self):
        return f"LogicDense(in_dim={self.in_dim}, out_dim={self.out_dim})"


class LogicConv2d(nn.Module):
    """
    2D convolutional layer using binary trees of 2-input logic gates.

    Architecture:
    - Each kernel processes a receptive field using a binary tree structure
    - Tree depth d means 2^d leaf nodes combine inputs from the receptive field
    - Each tree node is a 2-input logic gate that combines its children's outputs

    For example, with tree_depth=2 and receptive_field_size=3:
    - 4 leaf gates select from the 3x3 receptive field
    - 2 gates combine pairs of leaf outputs
    - 1 gate combines to produce final kernel output

    The convolution slides this tree structure across spatial positions.

    Args:
        in_dim: Input spatial dimensions (H, W)
        channels: Number of input channels
        num_kernels: Number of output kernels (analogous to output channels)
        tree_depth: Depth of the binary tree (tree has 2^depth leaves)
        receptive_field_size: Spatial size (H, W) of receptive field
        stride: Convolution stride
        padding: Zero-padding applied before convolution
        connection_indices: List of index tensors for each tree level
        tree_lut_ids: List of LUT ID tensors for each tree level
    """

    def __init__(self, in_dim, channels, num_kernels, tree_depth,
                 receptive_field_size, stride, padding,
                 connection_indices, tree_lut_ids):
        super().__init__()
        self.in_dim = in_dim
        self.channels = channels
        self.num_kernels = num_kernels
        self.tree_depth = tree_depth
        self.receptive_field_size = receptive_field_size
        self.stride = stride
        self.padding = padding

        # Register connection indices and LUT IDs for each tree level
        for i, (conn_idx, lut_ids) in enumerate(zip(connection_indices, tree_lut_ids)):
            self.register_buffer(f'connection_indices_{i}', conn_idx)
            self.register_buffer(f'lut_ids_{i}', lut_ids)

        # Compute output spatial dimensions
        self.out_dim = tuple(
            (in_d + 2*padding - rf_size) // stride + 1
            for in_d, rf_size in zip(in_dim, receptive_field_size)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channels, H, W)
        Returns:
            Output tensor of shape (batch, num_kernels, H_out, W_out)
        """
        batch_size = x.shape[0]

        # Apply padding if needed
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
                mode='constant',
                value=0
            )

        # Level 0: Extract receptive field inputs using sliding window
        # connection_indices_0 shape: [2, num_kernels, num_positions, num_leaves, 3]
        # where last dim is (h, w, c)
        conn_0 = getattr(self, 'connection_indices_0')
        lut_0 = getattr(self, 'lut_ids_0')

        # Index into input: use advanced indexing
        # We need to extract x[batch, c, h, w] for each position
        indices_shape = conn_0.shape  # [2, num_kernels, num_positions, num_leaves, 3]

        # Extract h, w, c indices
        h_idx = conn_0[..., 0]  # shape: [2, num_kernels, num_positions, num_leaves]
        w_idx = conn_0[..., 1]
        c_idx = conn_0[..., 2]

        # Select from input using advanced indexing
        # x shape: [batch, channels, H, W]
        # We want: [batch, 2, num_kernels, num_positions, num_leaves]
        selected = x[:, c_idx, h_idx, w_idx]  # [batch, 2, K, P, L]

        # Apply LUTs at level 0
        # lut_0 shape: [num_leaves, num_kernels]
        a = selected[:, 0]  # [batch, K, P, L]
        b = selected[:, 1]

        # Reshape for vectorized application
        batch, K, P, L = a.shape
        a_flat = a.permute(0, 2, 1, 3).reshape(batch * P, K * L)  # [batch*P, K*L]
        b_flat = b.permute(0, 2, 1, 3).reshape(batch * P, K * L)
        lut_0_flat = lut_0.T.flatten()  # [K*L]

        result = apply_lut_vectorized(a_flat, b_flat, lut_0_flat)
        result = result.reshape(batch, P, K, L).permute(0, 2, 1, 3)  # [batch, K, P, L]

        # Process remaining tree levels
        for level in range(1, self.tree_depth + 1):
            conn_level = getattr(self, f'connection_indices_{level}')
            lut_level = getattr(self, f'lut_ids_{level}')

            # conn_level shape: [2, num_nodes_at_level]
            # We need to select from result along the last dimension
            selected = result[..., conn_level]  # [batch, K, P, 2, num_nodes]

            a = selected[..., 0, :]
            b = selected[..., 1, :]

            # Apply LUTs
            batch, K, P, N = a.shape
            a_flat = a.permute(0, 2, 1, 3).reshape(batch * P, K * N)
            b_flat = b.permute(0, 2, 1, 3).reshape(batch * P, K * N)
            lut_flat = lut_level.T.flatten()  # [K*N]

            result = apply_lut_vectorized(a_flat, b_flat, lut_flat)
            result = result.reshape(batch, P, K, N).permute(0, 2, 1, 3)  # [batch, K, P, N]

        # Reshape spatial dimension: [batch, K, P, 1] -> [batch, K, H_out, W_out]
        result = result.squeeze(-1)  # [batch, K, P]
        result = result.view(batch, self.num_kernels, *self.out_dim)

        return result

    def __repr__(self):
        return (f"LogicConv2d(in_dim={self.in_dim}, channels={self.channels}, "
                f"num_kernels={self.num_kernels}, tree_depth={self.tree_depth}, "
                f"receptive_field_size={self.receptive_field_size})")

class OrPooling2d(torch.nn.Module):
    """OrPooling is actually just max pooling as all values are binary (0 or 1)."""

    def __init__(self, kernel_size, stride, padding=0):
        super(OrPooling2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """Pool the max value in the kernel."""
        assert x.dim() == 4, "Input tensor must be 4d"
        x = torch.nn.functional.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        return x    


if __name__ == "__main__":
    import numpy as np
    from torchvision import datasets, transforms

    model = torch.load("model-for-chang.pt", map_location="cpu", weights_only=False)
    print(model)

    dataset = datasets.MNIST('experiments/data-mnist', train=False, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    for x, y in dataloader:
        scores = model(x)
        pred = scores.argmax(dim=1)
        print(f"real labels: {y}")
        print(f"pred labels: {pred}")
        print(f"accuracy: {(pred == y).float().mean():.4f}")
        break

