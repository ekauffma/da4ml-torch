import torch
from da4ml.converter.plugin import DAISTracerPluginBase, _flatten_arr
from da4ml.trace import FixedVariableArray
from torch.fx import Node, Tracer

from .layers import _registered_modules


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
                    raise NotImplementedError(f'call_function is not supported: {target}')
                case 'call_method':
                    raise NotImplementedError(f'call_method is not supported: {target}')
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
