import torch
import numpy as np
from torchlogix.layers import LogicDense, LogicConv2d, GroupSum
import pytest
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, FixedVariableArrayInput
from da4ml.codegen import RTLModel
from pickle import Unpickler


def test_dense_layer():

    layer = LogicDense(in_dim=1024, out_dim=1024)

    model = torch.nn.Sequential(
        layer
    )
    model.eval()

    inp, out = trace_model(model, inputs=FixedVariableArrayInput((1, layer.in_dim)).quantize(0,1,1))

    comb = comb_trace(inp, out)

    # random boolean input
    data_in = np.random.randint(0, 2, (2**10, layer.in_dim)).astype(bool)
    print(f"data_in shape: {data_in.shape}, dtype: {data_in.dtype}")

    with torch.no_grad():
        torch_out = model(torch.from_numpy(data_in)).numpy()
    
    comb_out = comb.predict(data_in)

    assert np.array_equal(torch_out, comb_out), "Outputs do not match!"

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
def test_conv_layer_safe_configs(in_dim, channels, num_kernels, receptive_field_size, tree_depth):
    layer = LogicConv2d(
        in_dim=in_dim,
        channels=channels,
        num_kernels=num_kernels,
        receptive_field_size=receptive_field_size,
        tree_depth=tree_depth
    )

    model = torch.nn.Sequential(layer)
    model.eval()

    inp, out = trace_model(
        model,
        inputs=FixedVariableArrayInput((1, channels, *layer.in_dim)).quantize(0,1,1)
    )
    comb = comb_trace(inp, out)

    # Random boolean input
    data_in = np.random.randint(0, 2, (2**10, channels, *layer.in_dim)).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(data_in)).numpy()

    comb_out = comb.predict(data_in)

    # Flatten torch_out to match comb_out shape
    torch_out_flat = torch_out.reshape(torch_out.shape[0], -1)

    # Compare outputs
    assert np.array_equal(torch_out_flat, comb_out), "Outputs do not match!"

@pytest.mark.parametrize("k,in_dim", [
    (1, 64),
    (2, 64),
    (4, 64),
    (4, 128),
])
def test_group_sum_layer(k, in_dim):
    layer = GroupSum(k)
    model = torch.nn.Sequential(layer)
    model.eval()

    inp, out = trace_model(
        model,
        inputs=FixedVariableArrayInput((in_dim,)).quantize(0, 1, 1)
    )

    comb = comb_trace(inp, out)

    data_in = np.random.randint(0, 2, (2**10, in_dim)).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(data_in)).numpy()

    comb_out = comb.predict(data_in)
    torch_out = torch_out.reshape(comb_out.shape)

    assert np.array_equal(torch_out, comb_out), "GroupSum outputs do not match!"
