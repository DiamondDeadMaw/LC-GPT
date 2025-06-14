import torch
from torch import nn

class CenterNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5):
        """
        normalized_shape: int or list/tuple of ints, the last dimension(s) to normalize.
        eps: not used in CenterNorm but kept for API compatibility.
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.D = 1
        for d in self.normalized_shape:
            self.D *= d
        
        # learnable scale and bias, same shapes as LayerNorm
        self.gamma = nn.Parameter(torch.ones(*self.normalized_shape))
        self.beta  = nn.Parameter(torch.zeros(*self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: any shape [..., *normalized_shape]
        returns: same shape as x
        """
        # compute mean over the normalized dimensions
        # if 1D normalization: for multi‑dim, we need to mean over all last dims
        mean = x.mean(dim=-1, keepdim=True)  
        
        # center and scale by D/(D-1)
        centered = x - mean
        scaled  = centered * (self.D / (self.D - 1))
        
        # apply learnable affine
        return scaled * self.gamma + self.beta

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}"
