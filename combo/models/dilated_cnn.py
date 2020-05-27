from typing import List

import torch
import torch.nn as nn

from allennlp import common
from allennlp import nn as allen_nn


class DilatedCnnEncoder(nn.Module, common.FromParams):

    def __init__(self,
                 input_dim: int,
                 filters: List[int],
                 kernel_size: List[int],
                 stride: List[int],
                 padding: List[int],
                 dilation: List[int],
                 activations: List[allen_nn.Activation]):
        super().__init__()
        conv1d_layers = []
        input_dims = [input_dim] + filters[:-1]
        output_dims = filters
        for idx in range(len(activations)):
            conv1d_layers.append(nn.Conv1d(
                in_channels=input_dims[idx],
                out_channels=output_dims[idx],
                kernel_size=kernel_size[idx],
                stride=stride[idx],
                padding=padding[idx],
                dilation=dilation[idx]))
        self.conv1d_layers = nn.ModuleList(conv1d_layers)
        self.activations = activations
        assert len(self.activations) == len(self.conv1d_layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for layer, activation in zip(self.conv1d_layers, self.activations):
            x = activation(layer(x))
        return x
