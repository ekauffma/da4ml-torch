import torch
import numpy as np
from torchlogix.layers import LogicDense, LogicConv2d
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
    data_in = np.random.randint(0, 2, (2**10, layer.in_dim)).astype(np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(data_in)).numpy()
    
    comb_out = comb.predict(data_in)
    assert np.array_equal(torch_out, comb_out), "Outputs do not match!"


def test_conv_layer():
    layer = LogicConv2d(in_dim=28, channels=1,num_kernels=16, receptive_field_size=3, tree_depth=3)

    print(f"{layer.in_dim=}")

    model = torch.nn.Sequential(
        layer
    )
    model.eval()

    inp, out = trace_model(model, inputs=FixedVariableArrayInput((1, 1, *layer.in_dim)).quantize(0,1,1))

    comb = comb_trace(inp, out)

    # random boolean input
    data_in = np.random.randint(0, 2, (2**10, 1, *layer.in_dim)).astype(np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(data_in)).numpy()

    comb_out = comb.predict(data_in)
    torch_out = torch_out.reshape(comb_out.shape)

    assert np.array_equal(torch_out, comb_out), "Outputs do not match!"

def test_nonsequential_model():

    class NonSequentialModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = LogicDense(in_dim=64, out_dim=64)
            self.layer2 = LogicDense(in_dim=64, out_dim=64)

        def forward(self, x):
            out1 = self.layer1(x)
            out2 = self.layer2(x)
            return out1 * out2 # AND of the two layers

    model = NonSequentialModel()
    model.eval()

    inp, out = trace_model(model, inputs=FixedVariableArrayInput((1, 64)).quantize(0,1,1), framework='torch')

    comb = comb_trace(inp, out)

    data_in = np.random.randint(0, 2, (2**10, 64)).astype(np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(data_in)).numpy()

    comb_out = comb.predict(data_in)
    assert np.array_equal(torch_out, comb_out), "Outputs do not match!"
