from typing import Any, Dict

import torch
from torch import nn


class TinyPhaseGRU(nn.Module):
    """Lightweight pointwise P-arrival picker for [batch, time, ENZ] input."""

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 32,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "dropout": dropout,
        }
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        directions = 2 if bidirectional else 1
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * directions, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features, _ = self.gru(waveform)
        features = self.output_dropout(features)
        return self.classifier(features).squeeze(-1)


def save_checkpoint(
    path: str,
    model: TinyPhaseGRU,
    preprocessing: Dict[str, Any],
) -> None:
    """Save portable weights rather than pickling the complete model object."""
    torch.save(
        {
            "format_version": 1,
            "model_class": "TinyPhaseGRU",
            "model_config": dict(model.config),
            "model_state_dict": model.state_dict(),
            "preprocessing": dict(preprocessing),
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> TinyPhaseGRU:
    checkpoint = torch.load(path, map_location=device)
    model = TinyPhaseGRU(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model

