import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassificationModel(nn.Module):
    """Nonlinear binary classifier with multiple hidden layers and dropout."""

    def __init__(
        self,
        in_features: int,
        initialization: str = "random",
        dropout_rate: float = 0.0,  # <- Added
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError("Input dimension must be a positive integer.")

        self.fc1 = nn.Linear(in_features, 256, dtype=dtype)
        self.fc2 = nn.Linear(256, 64, dtype=dtype)
        self.output = nn.Linear(64, 1, dtype=dtype)

        self.dropout = nn.Dropout(p=dropout_rate)  # <- Added dropout module

        # Initialization logic
        if initialization == "zero":
            for layer in [self.fc1, self.fc2, self.output]:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        elif initialization == "random":
            pass  # Use PyTorch default
        else:
            raise ValueError(f"Unsupported initialization type: {initialization}")

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x).squeeze(-1)
        return x


class ClassificationModel(nn.Module):
    """Nonlinear binary classifier with multiple hidden layers and dropout."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        initialization: str = "random",
        dropout_rate: float = 0.0,  # <- Added
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError("Input dimension must be a positive integer.")

        self.fc1 = nn.Linear(in_features, 256, dtype=dtype)
        self.fc2 = nn.Linear(256, 64, dtype=dtype)
        self.output = nn.Linear(64, num_classes, dtype=dtype)

        self.dropout = nn.Dropout(p=dropout_rate)  # <- Added dropout module

        # Initialization logic
        if initialization == "zero":
            for layer in [self.fc1, self.fc2, self.output]:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        elif initialization == "random":
            pass  # Use PyTorch default
        else:
            raise ValueError(f"Unsupported initialization type: {initialization}")

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x