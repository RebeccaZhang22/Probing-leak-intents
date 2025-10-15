import torch
from torch import Tensor
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Linear classifier with optional dropout."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 2,
        initialization: str = "random",
        dropout_rate: float = 0.0,  # <- New argument
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        print("# Classes:", num_classes if num_classes > 2 else 1)

        self.dropout = nn.Dropout(p=dropout_rate)  # <- Added dropout layer

        self.in_features = in_features

        self.linear = torch.nn.Linear(
            in_features,
            num_classes if num_classes > 2 else 1,
            dtype=dtype,
        )

        # Initialization logic
        if initialization == "zero":
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        elif initialization == "random":
            pass  # use PyTorch default
        else:
            raise ValueError(f"Unsupported initialization type: {initialization}")

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        return self.linear(x).squeeze(-1)