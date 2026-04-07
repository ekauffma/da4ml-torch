import torch
import torch.nn.functional as F
from da4ml.converter.plugin import DAISTracerPluginBase, _flatten_arr
from da4ml.trace import FixedVariableArray
from torch.fx import Node, Tracer

from .layers import _registered_modules

SUPPORTED_METHODS = {
    'apply',
    'as_new', 
    'flatten', 
    'from_kif', 
    'from_lhs', 
    'matmul', 
    'quantize', 
    'ravel', 
    'relu', 
    'reshape', 
    'rmatmul', 
    'to_bool', 
    'transpose',
}

SUPPORTED_FUNCTIONS = {
    'relu',
    'reshape',
    'flatten',
    'matmul',
}

OPERATOR_MAP = {
    "mul": "__mul__",
    "add": "__add__",
    "sub": "__sub__",
    "and_": "__and__",
    "or_": "__or__",
}

PASSTHROUGH_FUNCTIONS = {
    "getattr",
}

class DATracer(Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str):
        if type(m) in _registered_modules:
            return True
        return super().is_leaf_module(m, module_qualified_name)


class TorchParser(DAISTracerPluginBase):
    def trace(
        self,
        verbose: bool = False,
        inputs: tuple[FixedVariableArray, ...] | FixedVariableArray | None = None,
        inputs_kif: tuple[int, int, int] | None = None,
        dump: bool = False,
    ):
        assert inputs is not None
        if isinstance(inputs, FixedVariableArray):
            inputs = (inputs,)
        self.model: torch.nn.Module
        tracer = DATracer()
        graph = tracer.trace(self.model)
        modules = dict(self.model.named_modules())
        env: dict[str, FixedVariableArray] = {}
        inp_nodes = [n for n in graph.nodes if n.op == 'placeholder']
        out_nodes = [n for n in graph.nodes if n.op == 'output']
        assert len(out_nodes) == 1, f'only one output node is supported, but found {len(out_nodes)}'
        assert len(inputs) == len(inp_nodes), (
            f'inputs length {len(inputs)} does not match with graph input length {len(inp_nodes)}'
        )
        for node, inp in zip(inp_nodes, inputs):
            env[node.name] = inp

        for node in graph.nodes:
            args: tuple[Node, ...] = node.args  # type: ignore
            kwargs: dict[str, Node] = node.kwargs  # type: ignore
            target: str = node.target  # type: ignore
            match node.op:
                case 'call_module':
                    module = modules[target]
                    assert type(module) in _registered_modules, f'{type(module)} is not supported'
                    replay_cls = _registered_modules[type(module)]
                    replay = replay_cls(module)
                    _args = tuple(env[n.name] for n in args)
                    _lwargs = {k: env[v.name] for k, v in kwargs.items()}
                    env[node.name] = replay(*_args, **_lwargs)
                case 'call_function':
                    _args = tuple(env[n.name] if isinstance(n, Node) else n for n in args)
                    _kwargs = {k: env[v.name] if isinstance(v, Node) else v for k, v in kwargs.items()}
                    first_fva_arg = next((a for a in _args if isinstance(a, FixedVariableArray)), None)
                    op_name = target.__name__
                    if op_name in PASSTHROUGH_FUNCTIONS:
                        env[node.name] = target(*_args, **_kwargs)
                        continue
                    if op_name in OPERATOR_MAP:
                        if first_fva_arg is None:
                            env[node.name] = target(*_args, **_kwargs)
                            continue
                        method_name = OPERATOR_MAP[op_name]
                        if not hasattr(first_fva_arg, method_name):
                            raise NotImplementedError(
                                f"Operator '{op_name}' not implemented for FixedVariableArray"
                            )
                        method = getattr(first_fva_arg, method_name)
                        env[node.name] = method(_args[1])
                        continue
                    if first_fva_arg is not None:
                        if op_name not in SUPPORTED_FUNCTIONS:
                            raise NotImplementedError(
                                f"Function '{op_name}' not supported"
                            )
                        if not hasattr(first_fva_arg, op_name):
                            raise NotImplementedError(
                                f"Function '{op_name}' declared supported but not implemented"
                            )
                        method = getattr(first_fva_arg, op_name)
                        env[node.name] = method(*_args[1:], **_kwargs)
                    else:
                        env[node.name] = target(*_args, **_kwargs)
                case 'call_method':
                    _args = tuple(env[n.name] if isinstance(n, Node) else n for n in args)
                    _kwargs = {k: env[v.name] if isinstance(v, Node) else v for k, v in kwargs.items()}
                    obj = _args[0]
                    if isinstance(obj, FixedVariableArray):
                        if target not in SUPPORTED_METHODS:
                            raise NotImplementedError(
                                f"Method '{target}' is not supported for FixedVariableArray"
                            )
                        if not hasattr(obj, target):
                            raise NotImplementedError(
                                f"Method '{target}' declared supported but not implemented"
                            )
                        if target == "transpose": # handle PyTorch transpose API difference
                            if len(_args) == 3:
                                dim0, dim1 = _args[1], _args[2]
                                axes = list(range(obj.ndim))
                                axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
                                env[node.name] = obj.transpose(tuple(axes))
                                continue
                            env[node.name] = obj.transpose(*_args[1:], **_kwargs)
                            continue
                    method = getattr(obj, target)
                    env[node.name] = method(*_args[1:], **_kwargs)
                case 'get_attr':
                    attr = getattr(self.model, target)
                    if isinstance(attr, torch.Tensor):
                        attr = attr.detach().cpu().numpy()
                    env[node.name] = attr
                case 'placeholder':
                    pass
                case 'output':
                    pass
                case _:
                    raise NotImplementedError(f'unknown node op: {node.op}')

        inp_tensors = tuple(env[n.name] for n in inp_nodes)
        out_tensors = tuple(env[str(out_name)] for out_name in out_nodes[0].args)
        if not dump:
            return _flatten_arr(inp_tensors), _flatten_arr(out_tensors)
        return env
