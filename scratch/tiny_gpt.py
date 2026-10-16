"""A causal byte-level decoder with explicit masked attention and MLP blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 257  # 0 is reserved; byte b is token b + 1.
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.register_buffer("future_mask", torch.triu(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool), diagonal=1
        ), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)
        shape = (batch, time, self.n_head, self.head_dim)
        q = q.reshape(shape).transpose(1, 2)
        k = k.reshape(shape).transpose(1, 2)
        v = v.reshape(shape).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) * self.head_dim ** -0.5
        scores = scores.masked_fill(self.future_mask[:time, :time], torch.finfo(scores.dtype).min)
        weights = self.attn_drop(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(batch, time, width)
        return self.resid_drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.n_embd)
        self.position = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.norm = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        _, time = tokens.shape
        if time > self.config.block_size:
            raise ValueError("sequence exceeds block size")
        positions = torch.arange(time, device=tokens.device)
        x = self.drop(self.token(tokens) + self.position(positions))
        logits = self.head(self.norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(prefix[:, -self.config.block_size:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            prefix = torch.cat((prefix, torch.multinomial(probs, 1)), dim=1)
        return prefix


def encode(text: str) -> list[int]:
    return [byte + 1 for byte in text.encode("utf-8")]


def decode(tokens: list[int]) -> str:
    return bytes(max(0, token - 1) for token in tokens if token > 0).decode(
        "utf-8", errors="replace"
    )
