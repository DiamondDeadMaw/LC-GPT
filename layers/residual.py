import torch
from torch import nn

class WeightedResidual(nn.Module):
    def __init__(self, fn: nn.Module, channels: int, init_alpha: float = 0.1, max_alpha: float = 0.2, clamp_alpha: bool = True):
        """
        y = x + alpha * fn(x)

        Args:
            fn: the residual function f(x).
            channels: number of channels in x (size of the last dim).
            init_alpha: initial value for each element of alpha.
        """
        super().__init__()
        self.fn = fn
        self.clamp_alpha = clamp_alpha
        self.max_alpha = max_alpha

        # learnable per-channel scaling α, initialized to init_alpha
        self.alpha = nn.Parameter(torch.full((channels,), init_alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [..., channels]
        returns: same shape as x
        """
        # optionally clamp to [0, max_alpha] to enforce the Lipschitz bound
        if self.clamp_alpha:
            alpha = self.alpha.clamp(0.0, self.max_alpha)
        else:
            alpha = self.alpha

        # apply f(x)
        residual = self.fn(x)
        # scale per channel (broadcast over all other dims)
        return x + residual * alpha

    def extra_repr(self):
        return (f"fn={self.fn.__class__.__name__}, init_alpha={float(self.alpha.data.mean()):.3f}, "
                f"max_alpha={self.max_alpha}, clamp={self.clamp_alpha}")
