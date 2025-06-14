import torch
from torch import nn
from layers.attention import ScaledCosineSelfAttention
from layers.decoder import GPTDecoderBlock
from layers.normalization import CenterNorm

class GPT2(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_hidden: int,
        num_layers: int,
        vocab_size: int,
        res_init_alpha: float = 0.1,
        res_max_alpha: float = 0.2,
        attn_init_tau: float = 1.0,
        attn_init_nu: float = 1.0,
        attn_learnable_tau: bool = True,
        attn_learnable_nu: bool = True,
        dropout: float = 0.1,
    ):
        """
        GPT 2 decoder only Transformer.

        Args:
          dim: model hidden size
          num_heads: number of attention heads
          ff_hidden: hidden size of the 2 layer MLP (usually 4xdim)
          num_layers: number of stacked decoder blocks
          vocab_size: size of the output vocabulary
          res_init_alpha, res_max_alpha: weighted residual alpha init & clamp
          attn_*: init/learnable flags for tau and nu in attention
          dropout: dropout inside FeedForward
        """
        super().__init__()
        self.dim = dim

        # stack of decoder blocks
        self.blocks = nn.ModuleList([
            GPTDecoderBlock(
                dim=dim,
                num_heads=num_heads,
                ff_hidden=ff_hidden,
                attn_init_tau=attn_init_tau,
                attn_init_nu=attn_init_nu,
                attn_learnable_tau=attn_learnable_tau,
                attn_learnable_nu=attn_learnable_nu,
                res_init_alpha=res_init_alpha,
                res_max_alpha=res_max_alpha,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # final normalization
        self.ln_f = CenterNorm(dim)

        # unembedding to vocab
        self.unembed = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, dim] = token_embed + pos_embed
        returns: [batch, seq_len, vocab_size] logits
        """
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.unembed(x)
        return logits
