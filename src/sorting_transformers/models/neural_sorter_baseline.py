"""Optional differentiable sorter baseline (not wired into build_model by default)."""

from torch import nn


class NeuralSorterBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError(
            "NeuralSorterBaseline is an optional stretch goal and is not implemented."
        )
