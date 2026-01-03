"""Butterfly-parameterized adapters for orthogonal subspace fine-tuning."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D


def _pair_indices(dim: int, stage: int):
    stride = 2 ** stage
    for i in range(0, dim, stride * 2):
        for j in range(stride):
            if i + j + stride < dim:
                yield i + j, i + j + stride


class ButterflyRotation(nn.Module):
    """Learnable butterfly rotation implemented via stages of 2x2 mixing."""

    def __init__(self, dim: int, stages: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.stages = stages or int(math.log2(dim))
        if 2 ** self.stages > dim:
            raise ValueError("Stages exceed dimension")
        self.angles = nn.ParameterList()
        for stage in range(self.stages):
            params = []
            for idx, _ in enumerate(_pair_indices(dim, stage)):
                params.append(nn.Parameter(torch.zeros(1)))
            self.angles.append(nn.Parameter(torch.zeros(len(params))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError("Input dimension mismatch")
        orig_shape = x.shape
        out = x.reshape(-1, self.dim)
        for stage, angles in enumerate(self.angles):
            out = out.clone()
            for idx, (i, j) in enumerate(_pair_indices(self.dim, stage)):
                theta = angles[idx]
                c, s = torch.cos(theta), torch.sin(theta)
                vi = out[:, i].clone()
                vj = out[:, j].clone()
                out[:, i] = c * vi - s * vj
                out[:, j] = s * vi + c * vj
        return out.reshape(orig_shape)


class ButterflyLinearAdapter(nn.Module):
    """Wraps a frozen linear layer with learnable butterfly rotation."""

    def __init__(self, linear: nn.Linear, stages: Optional[int] = None):
        super().__init__()
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False
        self.rotation = ButterflyRotation(linear.in_features, stages=stages)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        flat = input.reshape(-1, orig_shape[-1])
        rotated = self.rotation(flat).reshape(orig_shape)
        return self.linear(rotated)


class ButterflyConv1DAdapter(nn.Module):
    """Butterfly adapter for GPT-style Conv1D (linear) layers."""

    def __init__(self, conv: Conv1D, stages: Optional[int] = None):
        super().__init__()
        self.conv = conv
        for param in self.conv.parameters():
            param.requires_grad = False
        in_dim = conv.weight.size(0)
        self.rotation = ButterflyRotation(in_dim, stages=stages)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        flat = input.reshape(-1, orig_shape[-1])
        rotated = self.rotation(flat).reshape(orig_shape)
        return self.conv(rotated)
