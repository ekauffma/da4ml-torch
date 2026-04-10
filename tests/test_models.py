import torch
import numpy as np
from torchlogix.layers import LogicDense, LogicConv2d, GroupSum, OrPooling2d
import pytest
from da4ml.converter import trace_model
from da4ml.trace import comb_trace, FixedVariableArray, FixedVariableArrayInput
from da4ml.codegen import RTLModel
from da4ml.trace.fixed_variable import HWConfig
from pickle import Unpickler

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
 
def assert_comb_matches_model(
    model: torch.nn.Module,
    symbolic_shape: tuple[int, ...],
    data_in: np.ndarray,
    hwconf: HWConfig = HWConfig(1, -1, -1),
) -> None:
    """Trace model, run comb prediction, and assert outputs match PyTorch."""
    model.eval()
    for module in model.modules():
        if hasattr(module, 'set_export_mode'):
            module.set_export_mode(True)

    inp, out = trace_model(
        model,
        inputs=FixedVariableArrayInput(symbolic_shape).quantize(0, 1, 1),
        hwconf=hwconf,
        framework='torch',
    )
    comb = comb_trace(inp, out)
 
    with torch.no_grad():
        torch_out = model(torch.from_numpy(data_in)).detach().cpu().numpy()
 
    comb_out = np.asarray(comb.predict(data_in))
 
    np.testing.assert_array_equal(
        torch_out.reshape(torch_out.shape[0], -1),
        comb_out.reshape(comb_out.shape[0], -1),
    )


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class TinyDenseModel(torch.nn.Module):
    """Minimal dense model for fast sanity check."""
    def __init__(self):
        super().__init__()
        self.layer = LogicDense(in_dim=2, out_dim=2)

    def forward(self, x):
        return self.layer(x)


class DenseGroupSumModel(torch.nn.Module):
    """Three stacked dense layers followed by GroupSum."""
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
    """Two conv layers, flatten, dense, GroupSum."""
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
    """Conv -> OrPooling2d -> flatten -> dense -> GroupSum."""
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
    """Conv -> pool -> conv -> flatten -> dense -> GroupSum."""
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
    """Deep chain of dense layers to stress-test graph traversal."""
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
    """Single input fans out to two branches whose outputs are combined."""
    def __init__(self):
        super().__init__()
        self.input_layer = LogicDense(in_dim=64, out_dim=64)
        self.branch1 = LogicDense(in_dim=64, out_dim=64)
        self.branch2 = LogicDense(in_dim=64, out_dim=64)

    def forward(self, x):
        x = self.input_layer(x)
        return self.branch1(x) * self.branch2(x)


class FanOutGroupSumModel(torch.nn.Module):
    """FanOut followed by GroupSum."""
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
    """Skip connection: output is AND of layer output and original input."""
    def __init__(self):
        super().__init__()
        self.layer = LogicDense(in_dim=64, out_dim=64)

    def forward(self, x):
        return self.layer(x) * x


class DeepSkipModel(torch.nn.Module):
    """Three layers where layer1 output skips layer2 and merges at layer3."""
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
    """Conv -> flatten -> dense."""
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
    """Two independent inputs processed by separate layers, combined via AND."""
    def __init__(self):
        super().__init__()
        self.layer1 = LogicDense(in_dim=64, out_dim=64)
        self.layer2 = LogicDense(in_dim=64, out_dim=64)

    def forward(self, x1, x2):
        return self.layer1(x1) * self.layer2(x2)


class ChainOpsModel(torch.nn.Module):
    """
    Multiple call_method
    """
    def __init__(self):
        super().__init__()
        self.logic = LogicDense(64, 64)

    def forward(self, x):
        x = self.logic(x)
        x = x.reshape(x.shape[0], 8, 8)
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], -1)
        return x


class BroadcastModel(torch.nn.Module):
    """
    Tests implicit broadcasting, shape alignment, element-wise operations
    """
    def __init__(self):
        super().__init__()
        self.layer = LogicDense(64, 64)

    def forward(self, x):
        bias = torch.ones(64)
        return self.layer(x) * bias


class ScalarMulModel(torch.nn.Module):
    """
    Tests scalar multiplication within forward
    """
    def __init__(self):
        super().__init__()
        self.layer = LogicDense(64, 64)

    def forward(self, x):
        return self.layer(x) * 1


class ShapeAccessModel(torch.nn.Module):
    """
    Tests reshaping in forward function (getattr passthrough logic in parser)
    """
    def __init__(self):
        super().__init__()
        self.layer = LogicDense(64, 64)

    def forward(self, x):
        x = self.layer(x)
        b = x.shape[0]
        return x.reshape(b, -1)

# ---------------------------------------------------------------------------
# Parametrized model tests
# ---------------------------------------------------------------------------
 
@pytest.mark.parametrize("model_cls,symbolic_shape,data_shape", [
    (TinyDenseModel,       (1, 2),          (2**10, 2)),
    (DeepDenseModel,       (1, 64),         (2**10, 64)),
    (DenseGroupSumModel,   (1, 64),         (2**10, 64)),
    (ConvGroupSumModel,    (1, 1, 28, 28),  (2**10, 1, 28, 28)),
    (ConvPoolDenseModel,   (1, 1, 28, 28),  (2**10, 1, 28, 28)),
    (DoubleConvPoolModel,  (1, 1, 28, 28),  (2**10, 1, 28, 28)),
    (FanOutModel,          (1, 64),         (2**10, 64)),
    (FanOutGroupSumModel,  (1, 64),         (2**10, 64)),
    (ResidualModel,        (1, 64),         (2**10, 64)),
    (DeepSkipModel,        (1, 64),         (2**10, 64)),
    (MixedModel,           (1, 1, 28, 28),  (2**10, 1, 28, 28)),
    (ChainOpsModel,        (1, 64),         (2**10, 64)),
    (BroadcastModel,       (1, 64),         (2**10, 64)),
    (ScalarMulModel,       (1, 64),         (2**10, 64)),
    (ShapeAccessModel,     (1, 64),         (2**10, 64)),
])
def test_model_matches_comb_trace(model_cls, symbolic_shape, data_shape):
    """Parametrized test: verify comb trace matches PyTorch for each model."""
    torch.manual_seed(42)
    np.random.seed(42)
 
    model = model_cls()
    data_in = np.random.randint(0, 2, data_shape).astype(np.bool_)
    assert_comb_matches_model(model, symbolic_shape, data_in)


def test_multi_input_model_matches_comb_trace():
    """Multi-input model: two separate inputs combined via AND."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = MultiInputModel()
    model.eval()
    for module in model.modules():
        if hasattr(module, 'set_export_mode'):
            module.set_export_mode(True)

    inputs = (
        FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
        FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
    )
    inp, out = trace_model(model, inputs=inputs, framework='torch')
    comb = comb_trace(inp, out)

    data_in = tuple(
        np.random.randint(0, 2, (2**10, 64)).astype(np.bool_)
        for _ in inputs
    )
    with torch.no_grad():
        torch_out = model(*[torch.from_numpy(d) for d in data_in]).detach().cpu().numpy()

    comb_out = np.asarray(comb.predict(data_in))

    np.testing.assert_array_equal(
        torch_out.reshape(torch_out.shape[0], -1),
        comb_out.reshape(comb_out.shape[0], -1),
    )


# ---------------------------------------------------------------------------
# Standalone correctness tests
# ---------------------------------------------------------------------------
 
def test_wrong_input_shape_raises():
    """Passing wrong input shape should raise an exception."""
    model = torch.nn.Sequential(LogicDense(in_dim=64, out_dim=64))
    model.eval()
    for module in model.modules():
        if hasattr(module, 'set_export_mode'):
            module.set_export_mode(True)
    with pytest.raises(Exception):
        trace_model(
            model,
            inputs=FixedVariableArrayInput((1, 32)).quantize(0, 1, 1),
            framework='torch',
        )
 
 
def test_comb_predict_is_deterministic():
    """Running comb.predict twice on the same input should give identical results."""
    torch.manual_seed(0)
    np.random.seed(0)
 
    model = torch.nn.Sequential(LogicDense(in_dim=64, out_dim=64))
    model.eval()
    for module in model.modules():
        if hasattr(module, 'set_export_mode'):
            module.set_export_mode(True)
 
    inp, out = trace_model(
        model,
        inputs=FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
        framework='torch',
    )
    comb = comb_trace(inp, out)
 
    data_in = np.random.randint(0, 2, (2**10, 64)).astype(np.bool_)
    np.testing.assert_array_equal(comb.predict(data_in), comb.predict(data_in))
 
 
def test_boundary_inputs_all_zeros_and_ones():
    """All-zero and all-one inputs should produce matching outputs."""
    model = torch.nn.Sequential(LogicDense(in_dim=64, out_dim=64))
    model.eval()
    for module in model.modules():
        if hasattr(module, 'set_export_mode'):
            module.set_export_mode(True)

 
    inp, out = trace_model(
        model,
        inputs=FixedVariableArrayInput((1, 64)).quantize(0, 1, 1),
        framework='torch',
    )
    comb = comb_trace(inp, out)
 
    for data_in in [
        np.zeros((2**10, 64), dtype=np.bool_),
        np.ones((2**10, 64), dtype=np.bool_),
    ]:
        with torch.no_grad():
            torch_out = model(torch.from_numpy(data_in)).detach().cpu().numpy()
        comb_out = np.asarray(comb.predict(data_in))
        np.testing.assert_array_equal(
            torch_out.reshape(torch_out.shape[0], -1),
            comb_out.reshape(comb_out.shape[0], -1),
            err_msg=f"Boundary test failed for all-{'zeros' if data_in.mean() == 0 else 'ones'} input",
        )


# ---------------------------------------------------------------------------
# Hardware config variants
# ---------------------------------------------------------------------------

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
    torch.manual_seed(0)
    np.random.seed(0)

    model = torch.nn.Sequential(LogicDense(in_dim=64, out_dim=64))
    data_in = np.random.randint(0, 2, (2**10, 64)).astype(np.bool_)
    assert_comb_matches_model(model, (1, 64), data_in, hwconf=hwconf)
