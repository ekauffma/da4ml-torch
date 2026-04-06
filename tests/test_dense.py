import torch
import numpy as np
from torchlogix.layers import LogicDense, LogicConv2d
import pytest
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, FixedVariableArray, FixedVariableArrayInput
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

class FanOutModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = LogicDense(in_dim=64, out_dim=64)
        self.branch1 = LogicDense(in_dim=64, out_dim=64)
        self.branch2 = LogicDense(in_dim=64, out_dim=64)

    def forward(self, x):
        x = self.input_layer(x)
        return self.branch1(x) * self.branch2(x)

class ResidualModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = LogicDense(in_dim=64, out_dim=64)

    def forward(self, x):
        return self.layer(x) * x

class DeepSkipModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = LogicDense(in_dim=64, out_dim=64)
        self.layer2 = LogicDense(in_dim=64, out_dim=64)
        self.layer3 = LogicDense(in_dim=64, out_dim=64)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        return self.layer3(out2) * out1 # skip over layer 2

class MixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = LogicConv2d(in_dim=28, channels=1, num_kernels=16,
                                receptive_field_size=3, tree_depth=3)
        
        # figure out the flattened dim of the conv layer
        dummy = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            conv_out = self.conv(dummy)
        conv_out_flat = conv_out.reshape(conv_out.shape[0], -1)
        flattened_dim = conv_out_flat.shape[1]

        self.dense = LogicDense(in_dim=flattened_dim, out_dim=5408)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.dense(x)

class MultiInputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = LogicDense(in_dim=64, out_dim=64)
        self.layer2 = LogicDense(in_dim=64, out_dim=64)

    def forward(self, x1, x2):
        return self.layer1(x1) * self.layer2(x2)

@pytest.mark.parametrize("model_cls,inputs,data_shape", [
    (
        FanOutModel,
        FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
        (2**10, 64)
    ),
    (
        ResidualModel,
        FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
        (2**10, 64)
    ),
    (
        DeepSkipModel,
        FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
        (2**10, 64)
    ),
    (
        MixedModel,
        FixedVariableArrayInput((1, 1, 28, 28)).quantize(0, 1, 1),
        (2**10, 1, 28, 28)
    ),
    (
        MultiInputModel,
        [
            FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
            FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
        ],
        (2**10, 64)
    ),
])
def test_nonsequential_model(model_cls, inputs, data_shape):
    model = model_cls()
    model.eval()

    if isinstance(inputs, list):
        inp, out = trace_model(
            model,
            inputs=tuple(inputs),
            framework='torch'
        )
        comb = comb_trace(inp, out)
        data_in = tuple(
            np.random.randint(0, 2, data_shape).astype(np.float32)
            for _ in inputs
        )
        with torch.no_grad():
            torch_out = model(*[torch.from_numpy(d) for d in data_in]).numpy()
        comb_out = comb.predict(data_in)
    else:
        inp, out = trace_model(
            model,
            inputs=inputs,
            framework='torch'
        )
        comb = comb_trace(inp, out)
        data_in = np.random.randint(0, 2, data_shape).astype(np.float32)
        with torch.no_grad():
            torch_out = model(torch.from_numpy(data_in)).numpy()
        comb_out = comb.predict(data_in)

    torch_out = torch_out.reshape(comb_out.shape)
    assert np.array_equal(torch_out, comb_out), f"{model_cls.__name__}: outputs do not match!"
