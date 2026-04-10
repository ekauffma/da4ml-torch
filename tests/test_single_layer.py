import torch
import numpy as np
from torchlogix.layers import LogicDense, LogicConv2d, GroupSum, OrPooling2d
import pytest
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, FixedVariableArrayInput
from da4ml.codegen import RTLModel
from pickle import Unpickler

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def assert_comb_matches_model(
    model: torch.nn.Module,
    symbolic_shape: tuple[int, ...],
    data_in: np.ndarray,
) -> None:
    """Trace model, run comb prediction, and assert outputs match PyTorch."""
    model.eval()
    for module in model.modules():
        if hasattr(module, 'set_export_mode'):
            module.set_export_mode(True)

    inp, out = trace_model(model, inputs=FixedVariableArrayInput(symbolic_shape).quantize(0, 1, 1))
    comb = comb_trace(inp, out)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(data_in)).detach().cpu().numpy()

    comb_out = np.asarray(comb.predict(data_in))

    np.testing.assert_array_equal(
        torch_out.reshape(torch_out.shape[0], -1),
        comb_out.reshape(comb_out.shape[0], -1),
    )

# ---------------------------------------------------------------------------
# LogicDense
# ---------------------------------------------------------------------------

def test_dense_layer():
    """
    Test basic LogicDense layer: verify comb trace matches PyTorch output
    """
    torch.manual_seed(0)
    np.random.seed(0)
 
    layer = LogicDense(in_dim=1024, out_dim=1024)
    model = torch.nn.Sequential(layer)

    data_in = np.random.randint(0, 2, (2**10, layer.in_dim)).astype(np.int64)
 
    assert_comb_matches_model(model, (1, layer.in_dim), data_in)


# ---------------------------------------------------------------------------
# LogicConv2d
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "in_dim,channels,num_kernels,receptive_field_size,tree_depth", [
        (4, 1, 2, 2, 2),
        (4, 1, 4, 2, 2),
        (4, 2, 2, 2, 3),
        (4, 2, 4, 2, 3),
        (4, 4, 2, 2, 4),
        (4, 4, 4, 2, 4),
        (28, 1, 2, 2, 2),
        (28, 1, 4, 2, 2),
        (28, 2, 2, 2, 3),
        (28, 2, 4, 2, 3),
        (28, 4, 2, 2, 4),
        (28, 4, 4, 2, 4),
        ((14, 18), 1, 2, 2, 2),
        ((14, 18), 1, 4, 2, 2),
        ((14, 18), 2, 2, 2, 3),
        ((14, 18), 2, 4, 2, 3),
        ((14, 18), 4, 2, 2, 4),
        ((14, 18), 4, 4, 2, 4),
    ]
)
def test_conv_layer(in_dim, channels, num_kernels, receptive_field_size, tree_depth):
    """
    LogicConv2d with a range of configurations: verify comb trace matches PyTorch.
    """
    torch.manual_seed(0)
    np.random.seed(0)
 
    layer = LogicConv2d(
        in_dim=in_dim,
        channels=channels,
        num_kernels=num_kernels,
        receptive_field_size=receptive_field_size,
        tree_depth=tree_depth,
    )
    model = torch.nn.Sequential(layer)
    data_in = np.random.randint(0, 2, (2**10, channels, *layer.in_dim)).astype(np.bool_)
 
    assert_comb_matches_model(model, (1, channels, *layer.in_dim), data_in)


# ---------------------------------------------------------------------------
# GroupSum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k,in_dim", [
    (1, 64),
    (2, 64),
    (4, 64),
    (4, 128),
])
def test_group_sum_layer(k, in_dim):
    """
    GroupSum with various k values: verify comb trace matches PyTorch.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    layer = GroupSum(k)
    model = torch.nn.Sequential(layer)
    data_in = np.random.randint(0, 2, (2**10, in_dim)).astype(np.float32)

    assert_comb_matches_model(model, (in_dim,), data_in)


# ---------------------------------------------------------------------------
# OrPooling2d
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channels,h,w,kernel_size,stride,padding", [
    (1,  8,  8, 2, 2, 0),
    (1, 16, 16, 2, 2, 0),
    (2,  8,  8, 2, 2, 0),
    (4,  8,  8, 2, 2, 0),
    (1,  8,  8, 2, 1, 0),  # stride=1, overlapping windows
    (1,  8,  8, 2, 2, 1),  # with padding
    (1,  8,  8, 3, 1, 0),  # kernel_size=3
    (2, 14, 18, 2, 2, 0),  # non-square input
])
def test_or_pooling2d_layer(channels, h, w, kernel_size, stride, padding):
    """
    OrPooling2d with various configs: verify comb trace matches PyTorch.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    layer = OrPooling2d(kernel_size=kernel_size, stride=stride, padding=padding)
    model = torch.nn.Sequential(layer)
    data_in = np.random.randint(0, 2, (2**10, channels, h, w)).astype(np.bool_)

    assert_comb_matches_model(model, (1, channels, h, w), data_in)
