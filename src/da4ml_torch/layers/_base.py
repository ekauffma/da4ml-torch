from typing import Any

import numpy as np
import torch
from da4ml.trace import FixedVariable, FixedVariableArray

_registered_modules: dict = dict()


class ReplayMeta(type):
    def __new__(cls, name, bases, dct):
        new_cls = super().__new__(cls, name, bases, dct)
        for handle in new_cls.handles:  # type: ignore
            _registered_modules[handle] = new_cls
        return new_cls


class ReplayBase(metaclass=ReplayMeta):
    handles = ()

    def __init__(self, module: torch.nn.Module):
        assert type(module) in self.handles
        self.module: Any = module  # type: ignore

    def call(self, *args, **kwargs) -> tuple[FixedVariableArray, ...] | FixedVariableArray | FixedVariable: ...

    def __call__(self, *args, **kwargs) -> tuple[FixedVariableArray, ...] | FixedVariableArray:
        outputs = self.call(*args, **kwargs)
        if isinstance(outputs, FixedVariable):
            outputs = FixedVariableArray(np.array([outputs]))
        return outputs
