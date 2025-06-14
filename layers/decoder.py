import torch
from torch import nn
from feed_forward import FeedForward
from normalization import CenterNorm
from attention import ScaledCosineSelfAttention
from residual import WeightedResidual

class GPTDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_hidden: int,
        attn_init_tau: float = 1.0,
        attn_init_nu: float = 1.0,
        attn_learnable_tau: bool = True,
        attn_learnable_nu: bool = True,
        res_init_alpha: float = 0.1,
        res_max_alpha: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Self‐attention sublayer
        self.ln1 = CenterNorm(dim)
        self.attn = ScaledCosineSelfAttention(
            dim=dim,
            num_heads=num_heads,
            init_tau=attn_init_tau,
            init_nu=attn_init_nu,
            learnable_tau=attn_learnable_tau,
            learnable_nu=attn_learnable_nu,
        )
        self.wres1 = WeightedResidual(
            fn=self.attn,
            channels=dim,
            init_alpha=res_init_alpha,
            max_alpha=res_max_alpha,
            clamp_alpha=True
        )
        
        # Feed‐forward sublayer
        self.ln2 = CenterNorm(dim)
        self.ffn = FeedForward(dim=dim, hidden_dim=ff_hidden, dropout=dropout)
        self.wres2 = WeightedResidual(
            fn=self.ffn,
            channels=dim,
            init_alpha=res_init_alpha,
            max_alpha=res_max_alpha,
            clamp_alpha=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        
        # Self‐attention + residual
        a = self.ln1(x)
        x = self.wres1(a)   # applies x + alpha * attn(a)
        
        # Feed‐forward + residual
        m = self.ln2(x)
        x = self.wres2(m)   # applies x + alpha * ffn(m)
        
        return x
