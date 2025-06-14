import torch
from torch import nn
from torch.nn import Parameter
from masking import CausalMask

class ScaledCosineSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-8, init_tau: float = 1.0, init_nu: float = 1.0, learnable_tau: bool = True, learnable_nu: bool = True):
        """
        SCSA with causal masking

        Args:
          dim: total hidden dimension (will be split across heads)
          num_heads: number of attention heads
          eps: small constant to stabilize normalization
          init_tau: initial scaling before softmax
          init_nu:  initial output scaling after weighted sum
          learnable_tau: whether tau is a learnable Parameter
          learnable_nu:  whether nu is a learnable Parameter
        """
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        # Q, K, V projections
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)

        # scaling parameters
        if learnable_tau:
            self.tau = Parameter(torch.full((1,), init_tau))
        else:
            self.register_buffer("tau", torch.tensor(init_tau))
        if learnable_nu:
            self.nu = Parameter(torch.full((1,), init_nu))
        else:
            self.register_buffer("nu", torch.tensor(init_nu))

        # causal mask layer
        self.mask = CausalMask()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, dim]
        returns: [batch, seq_len, dim]
        """
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        # project and reshape into heads: [B, H, T, D]
        Q = self.wq(x).view(B, T, H, D).transpose(1, 2)
        K = self.wk(x).view(B, T, H, D).transpose(1, 2)
        V = self.wv(x).view(B, T, H, D).transpose(1, 2)

        # row‑wise l2 normalization (over D)
        Q = Q / (Q.norm(dim=-1, keepdim=True) + self.eps)
        K = K / (K.norm(dim=-1, keepdim=True) + self.eps)
        V = V / (V.norm(dim=-1, keepdim=True) + self.eps)

        # similarity scores: [B, H, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1))  
        # scale before softmax
        scores = scores * self.tau

        # apply causal mask
        scores = self.mask(scores)

        # attention weights and weighted sum
        P = torch.softmax(scores, dim=-1) # [B, H, T, T]
        out = torch.matmul(P, V) # [B, H, T, D]

        # combine heads
        out = out.transpose(1, 2).reshape(B, T, C) # [B, T, dim]

        # final output scaling
        return out * self.nu
