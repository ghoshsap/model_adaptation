"""Blockwise butterfly adapters (e.g., 256-dim blocks for GPT-style layers)."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from .butterfly_rotation_adapter import ButterflyRotation


class BlockwiseButterflyRotation(nn.Module):
    """Applies independent butterfly rotations to fixed-size feature blocks."""

    def __init__(self, dim: int, block_size: int = 256, stages: Optional[int] = None):
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.dim = dim
        self.block_size = block_size
        self.block_starts: List[int] = []
        self.block_lengths: List[int] = []
        self.rotations = nn.ModuleList()
        for start in range(0, dim, block_size):
            length = min(block_size, dim - start)
            if length <= 1:
                # No useful rotation for singleton block; skip.
                continue
            block_stages = stages
            if block_stages is None:
                block_stages = max(1, int(math.log2(length)))
            self.block_starts.append(start)
            self.block_lengths.append(length)
            self.rotations.append(ButterflyRotation(length, stages=block_stages))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError("Input dimension mismatch")
        if not self.rotations:
            return x
        orig_shape = x.shape
        out = x.reshape(-1, self.dim)
        for start, length, rotation in zip(self.block_starts, self.block_lengths, self.rotations):
            segment = out[:, start : start + length]
            rotated = rotation(segment)
            out[:, start : start + length] = rotated
        return out.reshape(orig_shape)


class BlockwiseButterflyLinearAdapter(nn.Module):
    """Wraps a Linear layer with blockwise butterfly rotations."""

    def __init__(self, linear: nn.Linear, block_size: int = 256, stages: Optional[int] = None):
        super().__init__()
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False
        self.rotation = BlockwiseButterflyRotation(
            linear.in_features,
            block_size=block_size,
            stages=stages,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        flat = input.reshape(-1, orig_shape[-1])
        rotated = self.rotation(flat).reshape(orig_shape)
        return self.linear(rotated)


class BlockwiseButterflyConv1DAdapter(nn.Module):
    """GPT Conv1D variant of the blockwise butterfly wrapper."""

    def __init__(self, conv: Conv1D, block_size: int = 256, stages: Optional[int] = None):
        super().__init__()
        self.conv = conv
        for param in self.conv.parameters():
            param.requires_grad = False
        in_dim = conv.weight.size(0)
        self.rotation = BlockwiseButterflyRotation(
            in_dim,
            block_size=block_size,
            stages=stages,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        flat = input.reshape(-1, orig_shape[-1])
        rotated = self.rotation(flat).reshape(orig_shape)
        return self.conv(rotated)


def apply_blockwise_butterfly(
    model: nn.Module,
    target_modules,
    block_size: int = 256,
    stages: Optional[int] = None,
):
    """Recursively wrap target modules with blockwise butterfly adapters."""

    for name, module in model.named_children():
        matched = name in target_modules or any(name.endswith(t) for t in target_modules)
        if matched:
            if isinstance(module, nn.Linear):
                setattr(
                    model,
                    name,
                    BlockwiseButterflyLinearAdapter(module, block_size=block_size, stages=stages),
                )
                continue
            if isinstance(module, Conv1D):
                setattr(
                    model,
                    name,
                    BlockwiseButterflyConv1DAdapter(module, block_size=block_size, stages=stages),
                )
                continue
        apply_blockwise_butterfly(module, target_modules, block_size=block_size, stages=stages)
