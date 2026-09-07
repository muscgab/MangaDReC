#!/usr/bin/env python3
"""MPS/CUDA-compatible 14M semantic backbone for Fusion B v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FusionBStudentConfig:
    vocab_size: int = 6811
    character_count: int = 6807
    hidden_size: int = 384
    layers: int = 6
    attention_heads: int = 6
    ffn_size: int = 1536
    max_length: int = 256
    dropout: float = 0.1
    pad_token_id: int = 6807
    bos_token_id: int = 6808
    eos_token_id: int = 6809
    mask_token_id: int = 6810
    rope_base: float = 10000.0

    def to_dict(self) -> dict:
        return asdict(self)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Pair adjacent dimensions. This avoids complex tensors and is supported by MPS.
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    return rotated.flatten(-2)


class RoPESelfAttention(nn.Module):
    def __init__(self, config: FusionBStudentConfig):
        super().__init__()
        if config.hidden_size % config.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        self.heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads
        if self.head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = config.dropout
        inverse = 1.0 / (
            config.rope_base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inverse_frequency", inverse, persistent=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        position = torch.arange(length, device=x.device, dtype=torch.float32)
        phase = torch.outer(position, self.inverse_frequency.to(x.device))
        cos = phase.cos().to(x.dtype)[None, None, :, :]
        sin = phase.sin().to(x.dtype)[None, None, :, :]
        query = _apply_rope(query, cos, sin)
        key = _apply_rope(key, cos, sin)
        key_mask = attention_mask[:, None, None, :].to(torch.bool)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=key_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.output(attended)


class FusionBBlock(nn.Module):
    def __init__(self, config: FusionBStudentConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = RoPESelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.ffn_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_size, config.hidden_size),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x), attention_mask))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x * attention_mask.unsqueeze(-1).to(x.dtype)


class FusionBStudent(nn.Module):
    """Semantic pretraining form; visual adapter and action heads are added later."""

    def __init__(self, config: FusionBStudentConfig):
        super().__init__()
        self.config = config
        self.character_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        # 1 marks the first character of a new REC track. There is no SEP token.
        self.track_start_embedding = nn.Embedding(2, config.hidden_size)
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(FusionBBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.mlm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlm_norm = nn.LayerNorm(config.hidden_size)
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.apply(self._initialize)
        with torch.no_grad():
            self.character_embedding.weight[config.pad_token_id].zero_()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        track_starts: torch.Tensor,
        input_addition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.character_embedding(input_ids)
        x = x + self.track_start_embedding(track_starts.long())
        if input_addition is not None:
            x = x + input_addition
        x = self.input_dropout(self.input_norm(x))
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.final_norm(x)

    def masked_logits(self, hidden: torch.Tensor, masked_positions: torch.Tensor) -> torch.Tensor:
        selected = hidden[masked_positions]
        selected = self.mlm_norm(F.gelu(self.mlm_dense(selected)))
        return F.linear(selected, self.character_embedding.weight, self.mlm_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        track_starts: torch.Tensor,
        masked_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.encode(input_ids, attention_mask, track_starts)
        if masked_positions is None:
            return hidden
        return self.masked_logits(hidden, masked_positions)


def parameter_report(model: FusionBStudent) -> dict:
    total = sum(parameter.numel() for parameter in model.parameters())
    embedding = model.character_embedding.weight.numel()
    return {
        "total": total,
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "character_embedding": embedding,
        "non_embedding": total - embedding,
    }
