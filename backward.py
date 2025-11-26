"""
Backward Pass Kernels for Algebraic Transformer

Optimized Triton kernels for the backward pass of attention and related operations.
Implements memory-efficient gradient computation following Flash Attention principles.

Key insight: We recompute attention weights during backward pass rather than storing them,
trading compute for memory (critical for long sequences).
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple

from .fused_ops import (
    rational_sigmoid,
    rational_sigmoid_grad,
    rational_softmax_prob,
    rational_softmax_prob_and_grad,
    EPS,
    NEG_INF,
)


# =============================================================================
# BACKWARD KERNEL CONFIGURATIONS
# =============================================================================

ATTN_BWD_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
]


# =============================================================================
# ATTENTION BACKWARD - COMPUTE dQ
# =============================================================================

@triton.jit
def _compute_dq_block(
    # Forward tensors
    Q_ptr, K_ptr, V_ptr,
    # Gradient tensors
    dOut_ptr, dQ_ptr,
    # Saved tensors
    L_ptr,  # Normalization factors from forward
    # Dimensions
    seq_len, d_head, 
    # ALiBi
    alibi_slope,
    # Block info
    block_m_start,
    # Parameters
    scale,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Compute gradient w.r.t. Q for a block of rows.
    
    dQ = sum_j (dP_ij * K_j)
    
    where dP_ij is the gradient through the attention weights.
    """
    # Row range for this block
    m_range = block_m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < seq_len
    
    # Load Q block and dOut block
    q_ptrs = Q_ptr + m_range[:, None] * d_head + tl.arange(0, BLOCK_D)[None, :]
    dout_ptrs = dOut_ptr + m_range[:, None] * d_head + tl.arange(0, BLOCK_D)[None, :]
    
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    dout = tl.load(dout_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    
    # Load normalization factors
    L = tl.load(L_ptr + m_range, mask=m_mask, other=1.0).to(tl.float32)
    
    # Initialize dQ accumulator
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # First, we need to compute D = sum_j(dOut_ij * O_ij) = sum_j(dOut_ij * P_ij * V_j)
    # This is used in the gradient computation
    # Actually for rational softmax: we need sum_j(dOut * V * dP/dS)
    
    # Compute D = row-wise dot product of dOut and recomputed output
    D = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Determine iteration range
    if CAUSAL:
        max_block_n = block_m_start + BLOCK_M
    else:
        max_block_n = seq_len
    
    # First pass: compute D
    for block_n_start in range(0, max_block_n, BLOCK_N):
        n_range = block_n_start + tl.arange(0, BLOCK_N)
        n_mask = n_range < seq_len
        
        # Load K
        k_ptrs = K_ptr + n_range[:, None] * d_head + tl.arange(0, BLOCK_D)[None, :]
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        
        # Compute scores
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # ALiBi
        rel_pos = m_range[:, None] - n_range[None, :]
        alibi_bias = -tl.abs(rel_pos).to(tl.float32) * alibi_slope
        scores = scores + alibi_bias
        
        # Causal mask
        if CAUSAL:
            causal_mask = n_range[None, :] > m_range[:, None]
            scores = tl.where(causal_mask, NEG_INF, scores)
        
        # Sequence mask
        seq_mask = n_range[None, :] < seq_len
        scores = tl.where(seq_mask, scores, NEG_INF)
        
        # Compute normalized probabilities
        probs = rational_softmax_prob(scores)
        probs = probs / (L[:, None] + eps)
        probs = tl.where(seq_mask, probs, 0.0)
        if CAUSAL:
            probs = tl.where(causal_mask, 0.0, probs)
        
        # Load V
        v_ptrs = V_ptr + n_range[:, None] * d_head + tl.arange(0, BLOCK_D)[None, :]
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        
        # D = sum(dOut * P * V) = sum(dOut * (P @ V))
        pv = tl.dot(probs, v)
        D += tl.sum(dout * pv, axis=1)
    
    # Second pass: compute dQ
    for block_n_start in range(0, max_block_n, BLOCK_N):
        n_range = block_n_start + tl.arange(0, BLOCK_N)
        n_mask = n_range < seq_len
        
        # Load K, V
        k_ptrs = K_ptr + n_range[:, None] * d_head + tl.arange(0, BLOCK_D)[None, :]
        v_ptrs = V_ptr + n_range[:, None] * d_head + tl.arange(0, BLOCK_D)[None, :]
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        
        # Recompute scores and probs
        scores = tl.dot(q, tl.trans(k)) * scale
        
        rel_pos = m_range[:, None] - n_range[None, :]
        alibi_bias = -tl.abs(rel_pos).to(tl.float32) * alibi_slope
        scores = scores + alibi_bias
        
        if CAUSAL:
            causal_mask = n_range[None, :] > m_range[:, None]
            scores = tl.where(causal_mask, NEG_INF, scores)
        
        seq_mask = n_range[None, :] < seq_len
        scores = tl.where(seq_mask, scores, NEG_INF)
        
        # Compute prob and its derivative w.r.t scores
        probs, dprobs_dscores = rational_softmax_prob_and_grad(scores)
        probs = probs / (L[:, None] + eps)
        dprobs_dscores = dprobs_dscores / (L[:, None] + eps)
        
        # Apply masks
        probs = tl.where(seq_mask, probs, 0.0)
        dprobs_dscores = tl.where(seq_mask, dprobs_dscores, 0.0)
        if CAUSAL:
            probs = tl.where(causal_mask, 0.0, probs)
            dprobs_dscores = tl.where(causal_mask, 0.0, dprobs_dscores)
        
        # Compute dS = dOut @ V^T * dP/dS - P * D (simplified softmax-like gradient)
        # Actually for our rational softmax:
        # dL/dS_ij = dL/dP_ij * dP_ij/dS_ij
        # where dL/dP_ij = dOut_i . V_j (dot product)
        
        dout_v = tl.dot(dout, tl.trans(v))  # [BLOCK_M, BLOCK_N]
        
        # The gradient of normalized probs w.r.t. unnormalized scores:
        # dP_norm/dS = dP_unnorm/dS / L - P_norm * (sum over j of dP_unnorm/dS) / L
        # This is similar to softmax gradient
        
        # Simplified: dS = (dOut @ V^T) * dP/dS - P * D / L
        # where D = sum(dOut * P * V)
        dS = dout_v * dprobs_dscores - probs * (D[:, None] / (L[:, None] + eps))
        
        # dQ += dS @ K
        dq += tl.dot(dS.to(k.dtype) * scale, k)
    
    # Store dQ
    dq_ptrs = dQ_ptr + m_range[:, None] * d_head + tl.arange(0, BLOCK_D)[None, :]
    tl.store(dq_ptrs, dq.to(dQ_ptr.dtype.element_ty), mask=m_mask[:, None])


@triton.autotune(configs=ATTN_BWD_CONFIGS, key=['seq_len', 'd_head'])
@triton.jit
def rational_attention_bwd_dq_kernel(
    # Forward tensors
    Q_ptr, K_ptr, V_ptr,
    # Gradient tensors
    dOut_ptr, dQ_ptr,
    # Saved tensors
    L_ptr,
    # ALiBi
    alibi_slopes_ptr,
    # Strides
    stride_b, stride_h, stride_t, stride_d,
    stride_lb, stride_lh,
    # Dimensions
    batch_size, n_heads, seq_len, d_head,
    # Parameters
    scale,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Kernel to compute dQ."""
    block_m_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    
    batch_idx = batch_head_idx // n_heads
    head_idx = batch_head_idx % n_heads
    
    block_m_start = block_m_idx * BLOCK_M
    if block_m_start >= seq_len:
        return
    
    alibi_slope = tl.load(alibi_slopes_ptr + head_idx)
    
    # Compute base pointers
    base_offset = batch_idx * stride_b + head_idx * stride_h
    Q_base = Q_ptr + base_offset
    K_base = K_ptr + base_offset  
    V_base = V_ptr + base_offset
    dOut_base = dOut_ptr + base_offset
    dQ_base = dQ_ptr + base_offset
    L_base = L_ptr + batch_idx * stride_lb + head_idx * stride_lh
    
    BLOCK_D = d_head
    
    _compute_dq_block(
        Q_base, K_base, V_base,
        dOut_base, dQ_base,
        L_base,
        seq_len, d_head,
        alibi_slope,
        block_m_start,
        scale,
        eps,
        BLOCK_M, BLOCK_N, BLOCK_D,
        CAUSAL,
    )


# =============================================================================
# ATTENTION BACKWARD - COMPUTE dK, dV
# =============================================================================

@triton.jit
def _compute_dkv_block(
    # Forward tensors
    Q_ptr, K_ptr, V_ptr,
    # Gradient tensors
    dOut_ptr, dK_ptr, dV_ptr,
    # Saved tensors
    L_ptr,
    # Dimensions
    seq_len, d_head,
    # ALiBi  
    alibi_slope,
    # Block info
    block_n_start,
    # Parameters
    scale,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Compute gradients w.r.t. K and V for a block of columns.
    
    dK_j = sum_i (dP_ij^T * Q_i) 
    dV_j = sum_i (P_ij^T * dOut_i)
    """
    n_range = block_n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < seq_len
    
    # Load K, V blocks for this column range
    k_ptrs = K_ptr + n_range[:, None] * d_head + tl.arange(0, BLOCK_D)[None, :]
    v_ptrs = V_ptr + n_range[:, None] * d_head + tl.arange(0, BLOCK_D)[None, :]
    
    k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
    
    # Initialize gradient accumulators
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    # Determine iteration range
    if CAUSAL:
        # For causal, only rows >= block_n_start can attend to these columns
        min_block_m = block_n_start
    else:
        min_block_m = 0
    
    # Iterate over Q rows
    for block_m_start in range(min_block_m, seq_len, BLOCK_M):
        m_range = block_m_start + tl.arange(0, BLOCK_M)
        m_mask = m_range < seq_len
        
        # Load Q, dOut for this row block
        q_ptrs = Q_ptr + m_range[:, None] * d_head + tl.arange(0, BLOCK_D)[None, :]
        dout_ptrs = dOut_ptr + m_range[:, None] * d_head + tl.arange(0, BLOCK_D)[None, :]
        
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        dout = tl.load(dout_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        
        # Load normalization factors
        L = tl.load(L_ptr + m_range, mask=m_mask, other=1.0).to(tl.float32)
        
        # Compute scores: Q @ K^T (but we need K @ Q^T for this)
        scores = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_M, BLOCK_N]
        
        # ALiBi
        rel_pos = m_range[:, None] - n_range[None, :]
        alibi_bias = -tl.abs(rel_pos).to(tl.float32) * alibi_slope
        scores = scores + alibi_bias
        
        # Causal mask: position j can only be attended by positions i >= j
        if CAUSAL:
            causal_mask = n_range[None, :] > m_range[:, None]
            scores = tl.where(causal_mask, NEG_INF, scores)
        
        seq_mask = (m_mask[:, None]) & (n_mask[None, :])
        scores = tl.where(seq_mask, scores, NEG_INF)
        
        # Compute probs and gradients
        probs, dprobs_dscores = rational_softmax_prob_and_grad(scores)
        probs = probs / (L[:, None] + eps)
        dprobs_dscores = dprobs_dscores / (L[:, None] + eps)
        
        probs = tl.where(seq_mask, probs, 0.0)
        dprobs_dscores = tl.where(seq_mask, dprobs_dscores, 0.0)
        if CAUSAL:
            probs = tl.where(causal_mask, 0.0, probs)
            dprobs_dscores = tl.where(causal_mask, 0.0, dprobs_dscores)
        
        # Compute D for this block
        pv = tl.dot(probs, v)  # [BLOCK_M, BLOCK_D]
        D = tl.sum(dout * pv, axis=1)  # [BLOCK_M]
        
        # dV = P^T @ dOut
        dv += tl.dot(tl.trans(probs), dout)
        
        # Compute dS for dK
        dout_v = tl.dot(dout, tl.trans(v))  # [BLOCK_M, BLOCK_N]
        dS = dout_v * dprobs_dscores - probs * (D[:, None] / (L[:, None] + eps))
        
        # dK = dS^T @ Q
        dk += tl.dot(tl.trans(dS.to(q.dtype) * scale), q)
    
    # Store gradients
    dk_ptrs = dK_ptr + n_range[:, None] * d_head + tl.arange(0, BLOCK_D)[None, :]
    dv_ptrs = dV_ptr + n_range[:, None] * d_head + tl.arange(0, BLOCK_D)[None, :]
    
    tl.store(dk_ptrs, dk.to(dK_ptr.dtype.element_ty), mask=n_mask[:, None])
    tl.store(dv_ptrs, dv.to(dV_ptr.dtype.element_ty), mask=n_mask[:, None])


@triton.autotune(configs=ATTN_BWD_CONFIGS, key=['seq_len', 'd_head'])
@triton.jit
def rational_attention_bwd_dkv_kernel(
    # Forward tensors
    Q_ptr, K_ptr, V_ptr,
    # Gradient tensors
    dOut_ptr, dK_ptr, dV_ptr,
    # Saved tensors
    L_ptr,
    # ALiBi
    alibi_slopes_ptr,
    # Strides
    stride_b, stride_h, stride_t, stride_d,
    stride_lb, stride_lh,
    # Dimensions
    batch_size, n_heads, seq_len, d_head,
    # Parameters
    scale,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Kernel to compute dK and dV."""
    block_n_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    
    batch_idx = batch_head_idx // n_heads
    head_idx = batch_head_idx % n_heads
    
    block_n_start = block_n_idx * BLOCK_N
    if block_n_start >= seq_len:
        return
    
    alibi_slope = tl.load(alibi_slopes_ptr + head_idx)
    
    base_offset = batch_idx * stride_b + head_idx * stride_h
    Q_base = Q_ptr + base_offset
    K_base = K_ptr + base_offset
    V_base = V_ptr + base_offset
    dOut_base = dOut_ptr + base_offset
    dK_base = dK_ptr + base_offset
    dV_base = dV_ptr + base_offset
    L_base = L_ptr + batch_idx * stride_lb + head_idx * stride_lh
    
    BLOCK_D = d_head
    
    _compute_dkv_block(
        Q_base, K_base, V_base,
        dOut_base, dK_base, dV_base,
        L_base,
        seq_len, d_head,
        alibi_slope,
        block_n_start,
        scale,
        eps,
        BLOCK_M, BLOCK_N, BLOCK_D,
        CAUSAL,
    )


# =============================================================================
# PYTHON INTERFACE
# =============================================================================

def rational_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,  # Forward output (for potential recomputation verification)
    L: torch.Tensor,  # Normalization factors from forward
    grad_output: torch.Tensor,
    alibi_slopes: torch.Tensor,
    scale: float,
    causal: bool = True,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for rational attention.
    
    Args:
        q, k, v: Forward inputs [B, H, T, D]
        output: Forward output [B, H, T, D] (unused, here for API consistency)
        L: Normalization factors [B, H, T]
        grad_output: Gradient w.r.t. output [B, H, T, D]
        alibi_slopes: ALiBi slopes [H]
        scale: Attention scale
        causal: Whether forward was causal
        eps: Numerical stability
    
    Returns:
        grad_q, grad_k, grad_v: Gradients w.r.t. inputs
    """
    assert q.is_cuda, "Input must be on CUDA"
    
    B, H, T, D = q.shape
    
    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    grad_output = grad_output.contiguous()
    L = L.contiguous()
    alibi_slopes = alibi_slopes.contiguous()
    
    # Allocate gradients
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    
    # Compute strides
    stride_b = q.stride(0)
    stride_h = q.stride(1)
    stride_t = q.stride(2)
    stride_d = q.stride(3)
    stride_lb = L.stride(0)
    stride_lh = L.stride(1)
    
    # Grid for dQ kernel
    BLOCK_M = 64
    BLOCK_N = 64
    num_m_blocks = triton.cdiv(T, BLOCK_M)
    num_n_blocks = triton.cdiv(T, BLOCK_N)
    
    # Compute dQ
    grid_dq = (num_m_blocks, B * H)
    rational_attention_bwd_dq_kernel[grid_dq](
        q, k, v,
        grad_output, grad_q,
        L,
        alibi_slopes,
        stride_b, stride_h, stride_t, stride_d,
        stride_lb, stride_lh,
        B, H, T, D,
        scale,
        eps,
        CAUSAL=causal,
    )
    
    # Compute dK, dV
    grid_dkv = (num_n_blocks, B * H)
    rational_attention_bwd_dkv_kernel[grid_dkv](
        q, k, v,
        grad_output, grad_k, grad_v,
        L,
        alibi_slopes,
        stride_b, stride_h, stride_t, stride_d,
        stride_lb, stride_lh,
        B, H, T, D,
        scale,
        eps,
        CAUSAL=causal,
    )
    
    return grad_q, grad_k, grad_v


# =============================================================================
# COMBINED AUTOGRAD FUNCTION
# =============================================================================

class RationalAttentionFunction(torch.autograd.Function):
    """
    Complete autograd function for rational attention.
    
    Combines forward and backward passes with memory-efficient implementation.
    """
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alibi_slopes: torch.Tensor,
        scale: float,
        causal: bool,
        eps: float,
    ) -> torch.Tensor:
        from .forward import rational_attention_forward
        
        output, L = rational_attention_forward(q, k, v, alibi_slopes, scale, causal, eps)
        
        # Save for backward
        ctx.save_for_backward(q, k, v, output, L, alibi_slopes)
        ctx.scale = scale
        ctx.causal = causal
        ctx.eps = eps
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, output, L, alibi_slopes = ctx.saved_tensors
        
        grad_q, grad_k, grad_v = rational_attention_backward(
            q, k, v, output, L, grad_output,
            alibi_slopes,
            ctx.scale,
            ctx.causal,
            ctx.eps,
        )
        
        return grad_q, grad_k, grad_v, None, None, None, None


def rational_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alibi_slopes: torch.Tensor,
    scale: float,
    causal: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Memory-efficient rational attention with custom backward pass.
    
    This is the main entry point for attention computation in training.
    
    Args:
        q: Query tensor [B, H, T, D]
        k: Key tensor [B, H, T, D]
        v: Value tensor [B, H, T, D]
        alibi_slopes: ALiBi slopes [H]
        scale: Attention scale (typically 1/sqrt(d_head))
        causal: Apply causal masking
        eps: Numerical stability
    
    Returns:
        Attention output [B, H, T, D]
    """
    return RationalAttentionFunction.apply(q, k, v, alibi_slopes, scale, causal, eps)


# =============================================================================
# GRADIENT CHECKPOINTING SUPPORT
# =============================================================================

def rational_attention_with_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alibi_slopes: torch.Tensor,
    scale: float,
    causal: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Rational attention with activation recomputation.
    
    Uses torch.utils.checkpoint to trade compute for memory.
    Useful when training very long sequences where even saving L is expensive.
    """
    from torch.utils.checkpoint import checkpoint
    
    return checkpoint(
        RationalAttentionFunction.apply,
        q, k, v, alibi_slopes, scale, causal, eps,
        use_reentrant=False,
    )
