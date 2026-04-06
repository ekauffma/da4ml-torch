import torch
import numpy as np
from torchlogix.layers import LogicDense, LogicConv2d, GroupSum, OrPooling2d
import pytest
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, FixedVariableArray, FixedVariableArrayInput
from da4ml.codegen import RTLModel
from da4ml.trace.fixed_variable import HWConfig
from pickle import Unpickler

class TinyDenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = LogicDense(in_dim=2, out_dim=2)

    def forward(self, x):
        return self.layer(x)

class DenseGroupSumModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = LogicDense(in_dim=64, out_dim=64)
        self.layer2 = LogicDense(in_dim=64, out_dim=64)
        self.layer3 = LogicDense(in_dim=64, out_dim=64)
        self.groupsum = GroupSum(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.groupsum(x)

class ConvGroupSumModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = LogicConv2d(in_dim=28, channels=1, num_kernels=16,
                                 receptive_field_size=3, tree_depth=3)
        self.conv2 = LogicConv2d(in_dim=26, channels=1, num_kernels=16,
                                 receptive_field_size=3, tree_depth=3)
        self.dense = LogicDense(in_dim=9216, out_dim=4608)
        self.groupsum = GroupSum(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense(x)
        return self.groupsum(x)

class ConvPoolDenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = LogicConv2d(in_dim=28, channels=1, num_kernels=16,
                                receptive_field_size=3, tree_depth=3)
        self.pool = OrPooling2d(kernel_size=2, stride=2)

        # figure out flattened dim after pool
        dummy = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            conv_out = self.conv(dummy)
            pool_out = self.pool(conv_out)
        flattened_dim = pool_out.reshape(pool_out.shape[0], -1).shape[1]

        self.dense = LogicDense(in_dim=flattened_dim, out_dim=flattened_dim//2)
        self.groupsum = GroupSum(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense(x)
        return self.groupsum(x)

class DoubleConvPoolModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = LogicConv2d(in_dim=28, channels=1, num_kernels=16,
                                 receptive_field_size=3, tree_depth=3)
        self.pool = OrPooling2d(kernel_size=2, stride=2)
        self.conv2 = LogicConv2d(in_dim=13, channels=16, num_kernels=16,
                                 receptive_field_size=3, tree_depth=3)

        dummy = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            x = self.conv1(dummy)
            x = self.pool(x)
            x = self.conv2(x)
        flattened_dim = x.reshape(x.shape[0], -1).shape[1]

        self.dense = LogicDense(in_dim=flattened_dim, out_dim=flattened_dim//2)
        self.groupsum = GroupSum(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense(x)
        return self.groupsum(x)

class DeepDenseModel(torch.nn.Module):
    def __init__(self, depth=8):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [LogicDense(in_dim=64, out_dim=64) for _ in range(depth)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FanOutModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = LogicDense(in_dim=64, out_dim=64)
        self.branch1 = LogicDense(in_dim=64, out_dim=64)
        self.branch2 = LogicDense(in_dim=64, out_dim=64)

    def forward(self, x):
        x = self.input_layer(x)
        return self.branch1(x) * self.branch2(x)

class FanOutGroupSumModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = LogicDense(in_dim=64, out_dim=64)
        self.branch1 = LogicDense(in_dim=64, out_dim=64)
        self.branch2 = LogicDense(in_dim=64, out_dim=64)
        self.groupsum = GroupSum(1)

    def forward(self, x):
        x = self.input_layer(x)
        out = self.branch1(x) * self.branch2(x)
        return self.groupsum(out)

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
        TinyDenseModel,
        FixedVariableArrayInput((1, 2)).quantize(0, 1, 1),
        (2**10, 2)
    ),
    (
        DeepDenseModel,
        FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
        (2**10, 64)
    ),
    (
        DenseGroupSumModel,
        FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
        (2**10, 64)
    ),
    (
        ConvGroupSumModel,
        FixedVariableArrayInput((1, 1, 28, 28)).quantize(0, 1, 1),
        (2**10, 1, 28, 28)
    ),
    (
        ConvPoolDenseModel,
        FixedVariableArrayInput((1, 1, 28, 28)).quantize(0, 1, 1),
        (2**10, 1, 28, 28)
    ),
    (
        DoubleConvPoolModel,
        FixedVariableArrayInput((1, 1, 28, 28)).quantize(0, 1, 1),
        (2**10, 1, 28, 28)
    ),
    (
        FanOutModel,
        FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
        (2**10, 64)
    ),
    (
        FanOutGroupSumModel,
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
def test_model(model_cls, inputs, data_shape):
    np.random.seed(42)

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

def test_wrong_input_shape():
    """
    Tests whether receiving the wrong input shape raises an error
    """
    model = LogicDense(in_dim=64, out_dim=64)
    model = torch.nn.Sequential(model)
    model.eval()
    with pytest.raises(Exception):
        trace_model(
            model,
            inputs=FixedVariableArrayInput((1, 32)).quantize(0, 1, 1),  # wrong dim
            framework='torch'
        )

def test_determinism():
    """
    Tests whether running the same model twice on the same input results in the same output
    """
    model = LogicDense(in_dim=64, out_dim=64)
    model = torch.nn.Sequential(model)
    model.eval()

    inp, out = trace_model(
        model,
        inputs=FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
        framework='torch'
    )
    comb = comb_trace(inp, out)

    data_in = np.random.randint(0, 2, (2**10, 64)).astype(np.float32)
    out1 = comb.predict(data_in)
    out2 = comb.predict(data_in)
    assert np.array_equal(out1, out2), "comb.predict is not deterministic!"

def test_boundary_inputs():
    """
    Tests whether all-zero and all-one inputs behave correctly
    """
    model = torch.nn.Sequential(LogicDense(in_dim=64, out_dim=64))
    model.eval()

    inp, out = trace_model(
        model,
        inputs=FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
        framework='torch'
    )
    comb = comb_trace(inp, out)

    for data_in in [
        np.zeros((2**10, 64), dtype=np.float32),
        np.ones((2**10, 64), dtype=np.float32),
    ]:
        with torch.no_grad():
            torch_out = model(torch.from_numpy(data_in)).numpy()
        comb_out = comb.predict(data_in)
        torch_out = torch_out.reshape(comb_out.shape)
        assert np.array_equal(torch_out, comb_out), f"Boundary test failed for input with mean {data_in.mean()}"

@pytest.mark.parametrize("hwconf", [
    HWConfig(1, -1, -1),
    HWConfig(2, -1, -1),
    HWConfig(1, 4, -1),
    HWConfig(-1, -1, 10)
])
def test_hwconf_variants(hwconf):
    """
    Tests different hardware configuration. Default is (-1, -1, -1), meaning no constraints.
    Value 1: adder_size
    Value 2: carry_size:
    Value 3: latency_cutoff
    """
    model = torch.nn.Sequential(LogicDense(in_dim=64, out_dim=64))
    model.eval()
    inp, out = trace_model(
        model,
        inputs=FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
        hwconf=hwconf,
        framework='torch'
    )
    comb = comb_trace(inp, out)
    data_in = np.random.randint(0, 2, (2**10, 64)).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(data_in)).numpy()
    comb_out = comb.predict(data_in)
    torch_out = torch_out.reshape(comb_out.shape)
    assert np.array_equal(torch_out, comb_out)
