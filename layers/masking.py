import torch
from torch import nn

class CausalMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        attn_scores: Tensor of shape [batch, heads, seq_len, seq_len]
        """
        B, H, T, _ = attn_scores.shape

        # Create a causal mask: shape [1, 1, T, T]
        mask = torch.tril(torch.ones(T, T, device=attn_scores.device)).unsqueeze(0).unsqueeze(0)
        # Convert to float mask: 0 where allowed, -inf where disallowed
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

        return (attn_scores + mask)
