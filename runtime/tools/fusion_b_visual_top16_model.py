#!/usr/bin/env python3
"""Visual-semantic top-16 replacement head for the frozen Fusion B v1 task."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from fusion_b_student_model import FusionBStudent, FusionBStudentConfig, parameter_report


class FusionBVisualTop16(nn.Module):
    def __init__(self, config: FusionBStudentConfig):
        super().__init__()
        self.config = config
        self.backbone = FusionBStudent(config)
        self.visual_norm = nn.LayerNorm(120)
        self.visual_projection = nn.Linear(120, config.hidden_size)
        self.scalar_norm = nn.LayerNorm(6)
        self.scalar_projection = nn.Linear(6, config.hidden_size)
        self.candidate_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rank_bias = nn.Parameter(torch.zeros(16))
        self.keep_bias = nn.Parameter(torch.tensor(0.0))
        self.ctc_log_scale = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.visual_projection.weight, std=0.02)
        nn.init.zeros_(self.visual_projection.bias)
        nn.init.normal_(self.scalar_projection.weight, std=0.02)
        nn.init.zeros_(self.scalar_projection.bias)
        nn.init.normal_(self.candidate_projection.weight, std=0.02)

    def load_semantic_checkpoint(self, checkpoint: dict) -> None:
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.backbone.load_state_dict(state, strict=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        track_starts: torch.Tensor,
        visual: torch.Tensor,
        scalars: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_logp: torch.Tensor,
        keep_index: torch.Tensor,
        cell_mask: torch.Tensor,
        zero_visual: bool = False,
    ) -> torch.Tensor:
        addition = self.scalar_projection(self.scalar_norm(scalars))
        if not zero_visual:
            addition = addition + self.visual_projection(self.visual_norm(visual))
        hidden = self.backbone.encode(input_ids, attention_mask, track_starts, addition)
        cell_hidden = hidden[cell_mask]
        candidates = candidate_ids[cell_mask]
        logp = candidate_logp[cell_mask]
        keeps = keep_index[cell_mask]
        candidate_embedding = self.backbone.character_embedding(candidates)
        query = self.candidate_projection(cell_hidden)
        scores = (candidate_embedding * query[:, None, :]).sum(dim=-1) / math.sqrt(self.config.hidden_size)
        scores = scores + torch.exp(self.ctc_log_scale).clamp(max=10.0) * logp
        scores = scores + self.rank_bias
        scores.scatter_add_(1, keeps[:, None], self.keep_bias.expand(len(keeps), 1))
        return scores

    def report(self) -> dict:
        result = parameter_report(self.backbone)
        result["total_with_visual_head"] = sum(p.numel() for p in self.parameters())
        return result
