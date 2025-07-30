# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM layers module."""

from einops import rearrange
from torch import Tensor, nn


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b d n -> n b d")
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = rearrange(y, "n b d -> b d n")
        return y
