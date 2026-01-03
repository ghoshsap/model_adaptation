"""Butterfly adapters that freeze base weights and train only rotation parameters."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D


def _stage_pairs(dim: int, stage: int) -> List[Tuple[int, int]]:
    stride = 2 ** stage
    block = stride * 2
    pairs = []
    for start in range(0, dim, block):
        for offset in range(stride):
            i = start + offset
            j = i + stride
            if j < dim:
                pairs.append((i, j))
    return pairs


class ButterflyRotation(nn.Module):
    def __init__(self, dim: int, stages: Optional[int] = None):
        super().__init__()
        self.dim = dim
        max_stages = max(1, int(math.log2(dim)))
        self.stages = min(stages or max_stages, max_stages)
        self.angles = nn.ParameterList()
        self.stage_indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for stage in range(self.stages):
            pairs = _stage_pairs(dim, stage)
            if not pairs:
                break
            idx_i = torch.tensor([i for i, _ in pairs], dtype=torch.long)
            idx_j = torch.tensor([j for _, j in pairs], dtype=torch.long)
            self.register_buffer(f"idx_i_{stage}", idx_i)
            self.register_buffer(f"idx_j_{stage}", idx_j)
            self.stage_indices.append((idx_i, idx_j))
            self.angles.append(nn.Parameter(torch.zeros(len(pairs))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError("Input dimension mismatch")
        orig_shape = x.shape
        out = x.reshape(-1, self.dim)
        for (idx_i, idx_j), angles in zip(self.stage_indices, self.angles):
            device = out.device
            idx_i = idx_i.to(device)
            idx_j = idx_j.to(device)
            vi = out.index_select(1, idx_i)
            vj = out.index_select(1, idx_j)
            cos = torch.cos(angles).unsqueeze(0).to(device)
            sin = torch.sin(angles).unsqueeze(0).to(device)
            rotated_i = cos * vi - sin * vj
            rotated_j = sin * vi + cos * vj
            out = out.clone()
            out[:, idx_i] = rotated_i
            out[:, idx_j] = rotated_j
        return out.reshape(orig_shape)


class ButterflyLinearRotationAdapter(nn.Module):
    def __init__(self, linear: nn.Linear, stages: Optional[int] = None):
        super().__init__()
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False
        self.rotation = ButterflyRotation(linear.in_features, stages)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        flat = input.reshape(-1, orig_shape[-1])
        rotated = self.rotation(flat).reshape(orig_shape)
        return self.linear(rotated)


class ButterflyConv1DRotationAdapter(nn.Module):
    def __init__(self, conv: Conv1D, stages: Optional[int] = None):
        super().__init__()
        self.conv = conv
        for param in self.conv.parameters():
            param.requires_grad = False
        in_features = conv.weight.size(0)
        self.rotation = ButterflyRotation(in_features, stages)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        flat = input.reshape(-1, orig_shape[-1])
        rotated = self.rotation(flat).reshape(orig_shape)
        return self.conv(rotated)


def apply_butterfly_rotation(model: nn.Module, target_modules, stages: Optional[int] = None):
    for name, module in model.named_children():
        matched = name in target_modules or any(name.endswith(t) for t in target_modules)
        if matched:
            if isinstance(module, nn.Linear):
                setattr(model, name, ButterflyLinearRotationAdapter(module, stages))
                continue
            if isinstance(module, Conv1D):
                setattr(model, name, ButterflyConv1DRotationAdapter(module, stages))
                continue
        apply_butterfly_rotation(module, target_modules, stages)
