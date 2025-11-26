"""
Algebraic Transformer Language Model

A transformer using purely algebraic (rational) operations.
Optimized with Triton kernels for forward and backward passes.

Key differences from standard transformers:
- Rational softmax: σ(x)^4 / Σσ(x)^4 instead of exp-based
- Rational sigmoid in SwiGLU
- Mean-error normalization instead of RMSNorm/LayerNorm
- ALiBi positional encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

from .fused_ops import rational_softmax, rational_swiglu, mean_error_norm, EPS
from .forward import compute_alibi_slopes
from .backward import rational_attention


class RationalMR(nn.Module):
    """Mean-error normalization layer."""
    
    def __init__(self, d_model: int, eps: float = EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return mean_error_norm(x, self.weight, self.eps)
        else:
            mean_abs = x.abs().mean(dim=-1, keepdim=True)
            return (x / (mean_abs + self.eps)) * self.weight


class RationalSwiGLU(nn.Module):
    """SwiGLU FFN with rational sigmoid."""
    
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate_val = nn.Linear(d_model, 2 * d_ffn, bias=False)
        self.w_out = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gv = self.w_gate_val(x)
        gate, val = gv.chunk(2, dim=-1)
        
        if x.is_cuda:
            hidden = rational_swiglu(gate, val)
        else:
            abs_gate = gate.abs()
            sigma = 0.5 * (gate / (abs_gate + 1.0) + 1.0)
            hidden = gate * sigma * val
        
        return self.dropout(self.w_out(hidden))


class AlgebraicAttention(nn.Module):
    """Multi-head attention with rational softmax and ALiBi."""
    
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # ALiBi slopes
        slopes = compute_alibi_slopes(n_head)
        self.register_buffer("alibi_slopes", slopes)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv.unbind(0)
        
        if x.is_cuda:
            # Use optimized Triton kernel
            out = rational_attention(
                q, k, v,
                self.alibi_slopes,
                self.scale,
                causal=True,
                eps=EPS,
            )
        else:
            # CPU fallback
            out = self._cpu_attention(q, k, v, mask)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.out(out))
        
        return out
    
    def _cpu_attention(self, q, k, v, mask):
        """CPU fallback implementation."""
        B, H, T, D = q.shape
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # ALiBi
        pos = torch.arange(T, device=q.device)
        rel_pos = pos[:, None] - pos[None, :]
        alibi = -rel_pos.abs().float() * self.alibi_slopes.view(1, H, 1, 1)
        scores = scores + alibi
        
        # Causal mask
        causal = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(causal[None, None], -1e4)
        
        # Padding mask
        if mask is not None:
            pad = mask[:, None, None, :] == 0
            scores.masked_fill_(pad, -1e4)
        
        # Rational softmax
        probs = rational_softmax(scores, eps=EPS)
        
        return torch.matmul(probs, v)


class AlgebraicBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, d_model: int, n_head: int, d_ffn: int, dropout: float):
        super().__init__()
        self.attn = AlgebraicAttention(d_model, n_head, dropout)
        self.ffn = RationalSwiGLU(d_model, d_ffn, dropout)
        self.norm1 = RationalMR(d_model)
        self.norm2 = RationalMR(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class AlgebraicTransformerLM(nn.Module):
    """Algebraic Transformer Language Model."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        d_ffn: Optional[int] = None,
        block_size: int = 2048,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        if d_ffn is None:
            d_ffn = int(d_model * 8 / 3)
        
        self.block_size = block_size
        self.gradient_checkpointing = gradient_checkpointing
        self.n_layers = n_layers
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            AlgebraicBlock(d_model, n_head, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm = RationalMR(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.embed.weight = self.head.weight
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, '_scale_init'):
                std = 0.02 / math.sqrt(2 * self.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        x = self.embed(input_ids)
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x = block(x, attention_mask)
        
        x = self.norm(x)
        logits = self.head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_small_model(vocab_size: int = 32000, **kwargs) -> AlgebraicTransformerLM:
    """~125M parameters"""
    return AlgebraicTransformerLM(
        vocab_size=vocab_size,
        d_model=768,
        n_head=12,
        n_layers=12,
        **kwargs,
    )


def create_medium_model(vocab_size: int = 32000, **kwargs) -> AlgebraicTransformerLM:
    """~350M parameters"""
    return AlgebraicTransformerLM(
        vocab_size=vocab_size,
        d_model=1024,
        n_head=16,
        n_layers=24,
        **kwargs,
    )


def create_large_model(vocab_size: int = 32000, **kwargs) -> AlgebraicTransformerLM:
    """~760M parameters"""
    return AlgebraicTransformerLM(
        vocab_size=vocab_size,
        d_model=1536,
        n_head=24,
        n_layers=24,
        **kwargs,
    )
