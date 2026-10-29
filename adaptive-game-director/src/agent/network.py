"""Neural network architecture skeleton for the DQN Game Director.

ARCHITECTURE NOTE:
Since our RL Game Director observes aggregated player telemetry features (15-dim vector)
rather than high-dimensional screen pixels (Computer Vision), we use a Multi-Layer
Perceptron (MLP) Q-network rather than a convolutional network (CNN).
This keeps the director lightweight, computationally efficient, and fast to train.
"""

from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock base class when PyTorch is not yet installed
    class nn:  # type: ignore
        class Module:
            pass


class DDA_QNetwork(nn.Module):
    """Dueling Deep Q-Network for the AI Game Director.

    Decouples Q-value estimation into:
      - State-Value Stream V(s): "How balanced is the player's current Flow/Distress state?"
      - Action-Advantage Stream A(s, a): "How advantageous is each DDA modulation action?"
      - Aggregated via: Q(s, a) = V(s) + (A(s, a) - mean(A))

    This architectural split drastically accelerates DDA learning by assessing situation
    stability independently from individual intervention choices.
    """

    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 7,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        if not TORCH_AVAILABLE:
            self.feature_extractor = None
            self.value_stream = None
            self.advantage_stream = None
            return

        # Shared representation backbone (256-128 with LayerNorm)
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # State-Value Stream V(s): estimates baseline state pacing quality
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Action-Advantage Stream A(s, a): estimates relative benefit of each DDA action
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: "torch.Tensor") -> "torch.Tensor":
        """Forward pass computing Q-values via dueling aggregation.

        Args:
            state: Tensor of shape (batch_size, state_dim).

        Returns:
            Tensor of shape (batch_size, action_dim).
        """
        if not TORCH_AVAILABLE or self.feature_extractor is None:
            raise RuntimeError(
                "PyTorch is not installed. Please install torch to use DDA_QNetwork."
            )
        features = self.feature_extractor(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
