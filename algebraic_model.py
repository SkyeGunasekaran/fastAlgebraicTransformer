"""
Algebraic Transformer Language Model

A transformer architecture using purely algebraic (rational) operations.
Uses optimized Triton kernels for both forward and backward passes.

Key differences from standard transformers:
1. Rational softmax instead of exp-based softmax
2. Rational sigmoid in SwiGLU instead of SiLU
3. Mean-error normalization instead of RMSNorm/LayerNorm
4. ALiBi positional encoding (no learned positional embeddings)

Usage:
    from algebraic_model import AlgebraicTransformerLM
    
    model = AlgebraicTransformerLM(
        vocab_size=32000,
        d_model=768,
        n_head=12,
        n_layers=12,
    )
    
    # Training
    logits, loss = model(input_ids, labels=labels)
    loss.backward()
    
    # Inference
    generated = model.generate(prompt_ids, max_new_tokens=100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

# Import optimized kernels
from kernels import (
    rational_attention,
    rational_attention_with_recompute,
    rational_softmax,
    mean_error_norm,
    rational_swiglu,
    compute_alibi_slopes,
    fused_qkv_projection,
    EPS,
)


# =============================================================================
# MODULE DEFINITIONS
# =============================================================================

class RationalMR(nn.Module):
    """
    Mean-Error Normalization (Rational Mean Rescaling).
    
    Normalizes by dividing by the mean absolute value, then applies learnable scale.
    
    output = (x / (mean(|x|) + eps)) * weight
    
    This is a purely algebraic alternative to RMSNorm/LayerNorm.
    """
    
    def __init__(self, d_model: int, eps: float = EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return mean_error_norm(x, self.weight, self.eps)
        else:
            # CPU fallback
            magnitude = x.abs().mean(dim=-1, keepdim=True)
            return (x / (magnitude + self.eps)) * self.weight
    
    def extra_repr(self) -> str:
        return f'{self.weight.shape[0]}, eps={self.eps}'


class RationalSwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network with rational sigmoid.
    
    Uses rational sigmoid σ(x) = 0.5 * (x / (|x| + 1) + 1) instead of SiLU.
    
    output = W3(gate * σ(gate) * W_val(x))
    
    where [gate, val] = split(W_merged(x))
    """
    
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.w_merged = nn.Linear(d_model, 2 * d_ffn, bias=False)
        self.w3 = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Mark output projection for special initialization
        self.w3.SCALE_INIT = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = self.w_merged(x)
        gate, val = merged.chunk(2, dim=-1)
        
        if x.is_cuda:
            hidden = rational_swiglu(gate, val)
        else:
            # CPU fallback
            abs_gate = gate.abs()
            sigmoid_approx = 0.5 * (gate / (abs_gate + 1.0) + 1.0)
            hidden = gate * sigmoid_approx * val
        
        return self.dropout(self.w3(hidden))


class AlgebraicAttention(nn.Module):
    """
    Multi-Head Attention with rational softmax and ALiBi.
    
    Key features:
    1. Rational softmax: p_i = σ(s_i)^4 / Σ σ(s_j)^4
    2. ALiBi positional encoding: bias = -slope * |i - j|
    3. Memory-efficient implementation with custom backward pass
    """
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.dropout_p = dropout
        
        # Fused QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Mark output projection for special initialization
        self.out_proj.SCALE_INIT = True
        
        # Pre-compute ALiBi slopes
        alibi_slopes = compute_alibi_slopes(n_head)
        self.register_buffer("alibi_slopes", alibi_slopes)
        
        # Dropout for attention (applied in kernel for CUDA)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute QKV
        if x.is_cuda and T >= 64:  # Use fused projection for larger sequences
            qkv = fused_qkv_projection(x, self.qkv_proj.weight)
        else:
            qkv = self.qkv_proj(x)
        
        # Reshape: [B, T, 3*D] -> [B, T, 3, H, D_head] -> [3, B, H, T, D_head]
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Apply attention
        if x.is_cuda:
            # Use optimized Triton kernel
            out = rational_attention(
                q, k, v,
                self.alibi_slopes,
                self.scale,
                causal=True,
                eps=EPS,
            )
            
            # Apply dropout to output (since kernel doesn't support it internally yet)
            if self.training and self.dropout_p > 0:
                out = self.dropout(out)
        else:
            # CPU fallback with full attention matrix
            out = self._cpu_attention(q, k, v, attention_mask)
        
        # Reshape output: [B, H, T, D_head] -> [B, T, H, D_head] -> [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(out)
    
    def _cpu_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """CPU fallback for attention computation."""
        B, H, T, D = q.shape
        
        # Compute scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add ALiBi bias
        positions = torch.arange(T, device=q.device)
        relative_pos = positions[:, None] - positions[None, :]
        alibi_bias = -relative_pos.abs().float() * self.alibi_slopes.view(1, H, 1, 1)
        scores = scores + alibi_bias
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(causal_mask[None, None, :, :], -1e4)
        
        # Padding mask
        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :] == 0
            scores.masked_fill_(pad_mask, -1e4)
        
        # Rational softmax
        probs = rational_softmax(scores, eps=EPS)
        
        # Dropout
        if self.training:
            probs = self.dropout(probs)
        
        # Compute output
        return torch.matmul(probs, v)


class AlgebraicBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.
    
    x = x + Attention(Norm(x))
    x = x + FFN(Norm(x))
    """
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
    ):
        super().__init__()
        self.attn = AlgebraicAttention(d_model, n_head, dropout)
        self.ffn = RationalSwiGLU(d_model, d_ffn, dropout)
        self.norm1 = RationalMR(d_model)
        self.norm2 = RationalMR(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention with residual
        h = self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + h
        
        # FFN with residual
        h = self.ffn(self.norm2(x))
        x = x + h
        
        return x


# =============================================================================
# MAIN MODEL
# =============================================================================

class AlgebraicTransformerLM(nn.Module):
    """
    Algebraic Transformer Language Model.
    
    A decoder-only transformer using purely algebraic operations,
    optimized for training with custom Triton kernels.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_head: Number of attention heads
        n_layers: Number of transformer blocks
        d_ffn: FFN hidden dimension (default: 8/3 * d_model)
        block_size: Maximum sequence length
        dropout: Dropout probability
        gradient_checkpointing: Use activation checkpointing
        use_recompute_attention: Use attention recomputation (saves more memory)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        d_ffn: Optional[int] = None,
        block_size: int = 2048,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        use_recompute_attention: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        if d_ffn is None:
            d_ffn = int(d_model * 8 / 3)
        
        self.block_size = block_size
        self.gradient_checkpointing = gradient_checkpointing
        self.use_recompute_attention = use_recompute_attention
        self.n_layers = n_layers
        self.d_model = d_model
        
        # Token embedding (no positional embedding - using ALiBi)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            AlgebraicBlock(d_model, n_head, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.final_norm = RationalMR(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"AlgebraicTransformerLM: {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with scaled initialization."""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                # Scale down output projections by sqrt(2 * n_layers)
                std = 0.02 / math.sqrt(2 * self.n_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [B, T]
            labels: Target token IDs for loss computation [B, T]
            attention_mask: Padding mask [B, T] (1 = valid, 0 = pad)
        
        Returns:
            logits: Output logits [B, T, V]
            loss: Cross-entropy loss (if labels provided)
        """
        x = self.token_embedding(input_ids)
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x = block(x, attention_mask=attention_mask)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift for next-token prediction
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
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            idx: Starting token IDs [B, T]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
        
        Returns:
            Generated token IDs [B, T + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]  # Last token only
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(idx.size(0)):
                    for token_id in set(idx[i].tolist()):
                        logits[i, token_id] /= repetition_penalty
            
            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def configure_optimizers(
        self,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        device_type: str = "cuda",
    ) -> torch.optim.Optimizer:
        """
        Configure AdamW optimizer with weight decay.
        
        Separates parameters into those that should have weight decay
        (weights of linear layers) and those that shouldn't (biases, norms).
        """
        # Collect all parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Separate decay and no-decay parameters
        decay_params = []
        no_decay_params = []
        
        for pn, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)
        
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        # Use fused AdamW if available
        use_fused = device_type == "cuda" and torch.cuda.is_available()
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused,
        )
        
        return optimizer


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

def create_small_model(vocab_size: int = 32000, **kwargs) -> AlgebraicTransformerLM:
    """Create a small model (~125M params)."""
    return AlgebraicTransformerLM(
        vocab_size=vocab_size,
        d_model=768,
        n_head=12,
        n_layers=12,
        **kwargs,
    )


def create_medium_model(vocab_size: int = 32000, **kwargs) -> AlgebraicTransformerLM:
    """Create a medium model (~350M params)."""
    return AlgebraicTransformerLM(
        vocab_size=vocab_size,
        d_model=1024,
        n_head=16,
        n_layers=24,
        **kwargs,
    )


def create_large_model(vocab_size: int = 32000, **kwargs) -> AlgebraicTransformerLM:
    """Create a large model (~760M params)."""
    return AlgebraicTransformerLM(
        vocab_size=vocab_size,
        d_model=1536,
        n_head=24,
        n_layers=24,
        **kwargs,
    )


def create_xl_model(vocab_size: int = 32000, **kwargs) -> AlgebraicTransformerLM:
    """Create an XL model (~1.3B params)."""
    return AlgebraicTransformerLM(
        vocab_size=vocab_size,
        d_model=2048,
        n_head=32,
        n_layers=24,
        **kwargs,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Check for CUDA and Triton
    import sys
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU fallback")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    
    # Create model
    model = create_small_model().to(device)
    
    # Test forward pass
    batch_size = 2
    seq_len = 512
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    
    print("\nTesting forward pass...")
    logits, loss = model(input_ids, labels=labels)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test backward pass
    print("\nTesting backward pass...")
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    print(f"Total gradient norm: {total_grad_norm:.4f}")
    
    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, 32000, (1, 10), device=device)
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
    
    print("\nAll tests passed!")
