from pathlib import PurePath

import torch
from torch import Tensor
from torch.nn import functional, Linear, LSTM, Module


class TemporalModel(Module):
    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        num_of_layers: int,
        dropout: float,
        weight_file: None | PurePath = None,
    ):
        super().__init__()
        self.lstm = LSTM(
            feature_size, 
            hidden_size,
            num_of_layers,
            batch_first = True,
            dropout = dropout
        )
        if (weight_file):
            self.lstm.load_state_dict(torch.load(weight_file))
        self.hidden2label = Linear(hidden_size, 2)

    # Parameters:
    #   inputs: Shape=(batch, seq, feature_size)
    # Return: Shape=(batch, 2)
    def forward(self, inputs: Tensor) -> Tensor:
        a, _ = self.lstm(inputs) # => (batch, seq, hidden_size)
        a = self.hidden2label(a) # => (batch, seq, 2)
        # Pick only output of the last time step because intermediate steps are
        # not annotated.
        last_step = a[:, -1, :].squeeze(1) # => (batch, 2)
        return functional.log_softmax(last_step, 1) # activations => probabilities
