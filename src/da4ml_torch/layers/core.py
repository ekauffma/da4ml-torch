import numpy as np
import torch
from da4ml.trace import FixedVariableArray

from ._base import ReplayBase


class ReplayFlatten(ReplayBase):
    handles = (torch.nn.Flatten,)

    def call(self, input: FixedVariableArray):
        return np.ravel(input)[None]  # type: ignore
