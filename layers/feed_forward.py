import torch
import torch.nn as nn

def spectral_init(linear: nn.Linear):
    # standard Xavier-normal
    nn.init.xavier_normal_(linear.weight)

    # max eigenvalue and renormalize
    with torch.no_grad():
        sigma_max = torch.linalg.svdvals(linear.weight)[0]
        linear.weight.div_(sigma_max)
    # zero the bias
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

        # apply spectral init to both projections
        spectral_init(self.fc1)
        spectral_init(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
