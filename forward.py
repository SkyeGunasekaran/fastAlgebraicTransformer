"""
Forward Pass Kernels for Algebraic Transformer

Optimized Triton kernels for the forward pass of attention and related operations.
Implements memory-efficient attention similar to Flash Attention, but with rational softmax.
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple

from .fused_ops import (
    rational_sigmoid,
    rational_softmax_prob,
    EPS,
    NEG_INF,
)


# =============================================================================
# ATTENTION FORWARD KERNEL CONFIGURATIONS  
# =============================================================================

# Autotuning configs for different GPU architectures
ATTN_FWD_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
]


# =============================================================================
# CORE ATTENTION FORWARD KERNEL
# =============================================================================

@triton.jit
def _rational_attn_fwd_inner(
    # Pointers
    Q_block_ptr,
    K_block_ptr, 
    V_block_ptr,
    Out_block_ptr,
    L_ptr,  # Stores normalization factors for backward
    # Dimensions
    seq_len,
    d_head,
    # ALiBi
    alibi_slope,
    # Block indices
    block_m_start,
    # Constants
    scale,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Inner kernel for rational attention forward pass.
    
    Implements tiled computation:
    1. Load Q block
    2. Iterate over K,V blocks
    3. Compute attention scores with ALiBi
    4. Apply rational softmax (online normalization)
    5. Accumulate weighted values
    """
    # Initialize accumulators
    # acc: running weighted sum of values
    # l_sum: running sum of unnormalized probabilities (for normalization)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Load Q block - stays in registers throughout
    q = tl.load(Q_block_ptr)  # [BLOCK_M, BLOCK_D]
    
    # Compute row indices for this block
    m_range = block_m_start + tl.arange(0, BLOCK_M)
    
    # Determine iteration range based on causality
    if CAUSAL:
        # Only attend to positions <= current position
        max_block_n = block_m_start + BLOCK_M
    else:
        max_block_n = seq_len
    
    # Iterate over K,V blocks
    for block_n_start in range(0, max_block_n, BLOCK_N):
        # Load K block
        k = tl.load(K_block_ptr)  # [BLOCK_N, BLOCK_D]
        
        # Compute attention scores: Q @ K^T
        scores = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_M, BLOCK_N]
        
        # Compute column indices for this K block
        n_range = block_n_start + tl.arange(0, BLOCK_N)
        
        # Add ALiBi bias: -slope * |i - j|
        rel_pos = m_range[:, None] - n_range[None, :]
        alibi_bias = -tl.abs(rel_pos).to(tl.float32) * alibi_slope
        scores = scores + alibi_bias
        
        # Apply causal mask
        if CAUSAL:
            causal_mask = n_range[None, :] > m_range[:, None]
            scores = tl.where(causal_mask, NEG_INF, scores)
        
        # Apply sequence length mask
        seq_mask = n_range[None, :] < seq_len
        scores = tl.where(seq_mask, scores, NEG_INF)
        
        # Compute rational softmax probabilities (unnormalized)
        # p = σ(scores)^4
        probs = rational_softmax_prob(scores)  # [BLOCK_M, BLOCK_N]
        
        # Mask out invalid positions
        probs = tl.where(seq_mask, probs, 0.0)
        if CAUSAL:
            probs = tl.where(causal_mask, 0.0, probs)
        
        # Accumulate normalization factor
        l_sum += tl.sum(probs, axis=1)
        
        # Load V block and accumulate weighted sum
        v = tl.load(V_block_ptr)  # [BLOCK_N, BLOCK_D]
        acc += tl.dot(probs.to(v.dtype), v)
        
        # Advance K,V pointers
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # Normalize output
    l_sum = l_sum + eps
    acc = acc / l_sum[:, None]
    
    # Store output and normalization factor
    tl.store(Out_block_ptr, acc.to(Out_block_ptr.dtype.element_ty))
    tl.store(L_ptr + m_range, l_sum)


@triton.autotune(configs=ATTN_FWD_CONFIGS, key=['seq_len', 'd_head'])
@triton.jit
def rational_attention_fwd_kernel(
    # Input pointers
    Q_ptr, K_ptr, V_ptr,
    # Output pointers  
    Out_ptr,
    L_ptr,  # Normalization factors [B, H, T]
    # ALiBi slopes
    alibi_slopes_ptr,
    # Strides for Q
    stride_qb, stride_qh, stride_qt, stride_qd,
    # Strides for K
    stride_kb, stride_kh, stride_kt, stride_kd,
    # Strides for V
    stride_vb, stride_vh, stride_vt, stride_vd,
    # Strides for Out
    stride_ob, stride_oh, stride_ot, stride_od,
    # Strides for L
    stride_lb, stride_lh, stride_lt,
    # Dimensions
    batch_size, n_heads, seq_len, d_head,
    # Scale factor
    scale,
    # Constants
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Main kernel for rational attention forward pass.
    
    Grid: (num_m_blocks, batch_size * n_heads)
    """
    # Get program IDs
    block_m_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    
    # Decode batch and head indices
    batch_idx = batch_head_idx // n_heads
    head_idx = batch_head_idx % n_heads
    
    # Compute block start position
    block_m_start = block_m_idx * BLOCK_M
    
    # Skip if block is out of bounds
    if block_m_start >= seq_len:
        return
    
    # Load ALiBi slope for this head
    alibi_slope = tl.load(alibi_slopes_ptr + head_idx)
    
    # Compute base pointers for this batch/head
    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    out_base = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    l_base = L_ptr + batch_idx * stride_lb + head_idx * stride_lh
    
    # Create block pointers
    BLOCK_D = d_head  # Full head dimension
    
    Q_block_ptr = tl.make_block_ptr(
        base=q_base,
        shape=(seq_len, d_head),
        strides=(stride_qt, stride_qd),
        offsets=(block_m_start, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        base=k_base,
        shape=(seq_len, d_head),
        strides=(stride_kt, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=v_base,
        shape=(seq_len, d_head),
        strides=(stride_vt, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )
    
    Out_block_ptr = tl.make_block_ptr(
        base=out_base,
        shape=(seq_len, d_head),
        strides=(stride_ot, stride_od),
        offsets=(block_m_start, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    
    # Call inner kernel
    _rational_attn_fwd_inner(
        Q_block_ptr, K_block_ptr, V_block_ptr, Out_block_ptr,
        l_base,
        seq_len, d_head,
        alibi_slope,
        block_m_start,
        scale,
        eps,
        BLOCK_M, BLOCK_N, BLOCK_D,
        CAUSAL,
    )


# =============================================================================
# SIMPLIFIED NON-TILED ATTENTION (for shorter sequences)
# =============================================================================

@triton.jit
def rational_attention_fwd_simple_kernel(
    # Input pointers
    Q_ptr, K_ptr, V_ptr,
    # Output pointers
    Out_ptr, L_ptr,
    # ALiBi slopes
    alibi_slopes_ptr,
    # Dimensions
    seq_len, d_head, n_heads,
    # Strides
    stride_qb, stride_qh, stride_qt,
    stride_kb, stride_kh, stride_kt,
    stride_vb, stride_vh, stride_vt,
    stride_ob, stride_oh, stride_ot,
    stride_lb, stride_lh,
    # Scale
    scale,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simplified attention kernel for shorter sequences (seq_len <= 1024).
    Materializes full attention matrix but uses efficient memory access.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    row_idx = tl.program_id(2)
    
    if row_idx >= seq_len:
        return
    
    # Load ALiBi slope
    alibi_slope = tl.load(alibi_slopes_ptr + head_idx)
    
    # Compute pointers
    q_row = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh + row_idx * stride_qt
    k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    out_row = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh + row_idx * stride_ot
    
    # Load query vector
    q = tl.load(q_row + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < d_head, other=0.0)
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    l_sum = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over all key-value pairs
    for col_idx in range(seq_len):
        # Causal mask
        if col_idx > row_idx:
            continue
        
        # Load key
        k = tl.load(k_base + col_idx * stride_kt + tl.arange(0, BLOCK_SIZE), 
                    mask=tl.arange(0, BLOCK_SIZE) < d_head, other=0.0)
        
        # Compute attention score
        score = tl.sum(q * k) * scale
        
        # Add ALiBi bias
        score = score - alibi_slope * tl.abs(row_idx - col_idx).to(tl.float32)
        
        # Compute probability (unnormalized)
        prob = rational_softmax_prob(score)
        l_sum += prob
        
        # Load value and accumulate
        v = tl.load(v_base + col_idx * stride_vt + tl.arange(0, BLOCK_SIZE),
                    mask=tl.arange(0, BLOCK_SIZE) < d_head, other=0.0)
        acc += prob * v
    
    # Normalize and store
    acc = acc / (l_sum + eps)
    tl.store(out_row + tl.arange(0, BLOCK_SIZE), acc.to(Out_ptr.dtype.element_ty),
             mask=tl.arange(0, BLOCK_SIZE) < d_head)
    
    # Store normalization factor
    l_ptr = L_ptr + batch_idx * stride_lb + head_idx * stride_lh + row_idx
    tl.store(l_ptr, l_sum)


# =============================================================================
# PYTHON INTERFACE
# =============================================================================

def rational_attention_forward(
    q: torch.Tensor,  # [B, H, T, D]
    k: torch.Tensor,  # [B, H, T, D]
    v: torch.Tensor,  # [B, H, T, D]
    alibi_slopes: torch.Tensor,  # [H]
    scale: float,
    causal: bool = True,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for rational attention.
    
    Args:
        q: Query tensor [B, H, T, D]
        k: Key tensor [B, H, T, D]
        v: Value tensor [B, H, T, D]
        alibi_slopes: ALiBi slopes per head [H]
        scale: Attention scale factor (typically 1/sqrt(d_head))
        causal: Whether to apply causal masking
        eps: Numerical stability epsilon
    
    Returns:
        output: Attention output [B, H, T, D]
        L: Normalization factors [B, H, T] (saved for backward)
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA"
    assert q.shape == k.shape == v.shape, "Q, K, V must have same shape"
    
    B, H, T, D = q.shape
    
    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    alibi_slopes = alibi_slopes.contiguous()
    
    # Allocate outputs
    output = torch.empty_like(q)
    L = torch.empty((B, H, T), device=q.device, dtype=torch.float32)
    
    # Choose kernel based on sequence length
    if T <= 1024 and D <= 128:
        # Use simple kernel for short sequences
        BLOCK_SIZE = triton.next_power_of_2(D)
        grid = (B, H, T)
        
        rational_attention_fwd_simple_kernel[grid](
            q, k, v,
            output, L,
            alibi_slopes,
            T, D, H,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            L.stride(0), L.stride(1),
            scale,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Use tiled kernel for longer sequences
        # Determine grid size
        BLOCK_M = 64  # Will be autotuned
        num_m_blocks = triton.cdiv(T, BLOCK_M)
        grid = (num_m_blocks, B * H)
        
        rational_attention_fwd_kernel[grid](
            q, k, v,
            output, L,
            alibi_slopes,
            # Q strides
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            # K strides
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            # V strides
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            # Out strides
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            # L strides
            L.stride(0), L.stride(1), L.stride(2),
            # Dimensions
            B, H, T, D,
            scale,
            eps,
            CAUSAL=causal,
        )
    
    return output, L


# =============================================================================
# FUSED QKV PROJECTION + ATTENTION
# =============================================================================

@triton.jit
def fused_qkv_proj_kernel(
    # Input
    X_ptr,
    # Weight
    W_ptr,  # [3 * d_model, d_model] for fused QKV
    # Output
    QKV_ptr,
    # Dimensions
    batch_seq, d_model, d_out,
    # Strides
    stride_xb, stride_xd,
    stride_wd, stride_wo,
    stride_qb, stride_qo,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused QKV projection kernel.
    Computes: QKV = X @ W^T where W is [3*d_model, d_model]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block range
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Iterate over K dimension
    for k_start in range(0, d_model, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        
        # Load X block
        x_ptrs = X_ptr + rm[:, None] * stride_xb + rk[None, :] * stride_xd
        x_mask = (rm[:, None] < batch_seq) & (rk[None, :] < d_model)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load W block (transposed access)
        w_ptrs = W_ptr + rn[:, None] * stride_wo + rk[None, :] * stride_wd
        w_mask = (rn[:, None] < d_out) & (rk[None, :] < d_model)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(x, tl.trans(w))
    
    # Store result
    out_ptrs = QKV_ptr + rm[:, None] * stride_qb + rn[None, :] * stride_qo
    out_mask = (rm[:, None] < batch_seq) & (rn[None, :] < d_out)
    tl.store(out_ptrs, acc.to(QKV_ptr.dtype.element_ty), mask=out_mask)


def fused_qkv_projection(
    x: torch.Tensor,  # [B, T, D]
    weight: torch.Tensor,  # [3*D, D]
) -> torch.Tensor:
    """
    Fused QKV projection.
    
    Returns: QKV tensor [B, T, 3*D]
    """
    B, T, D = x.shape
    d_out = weight.shape[0]  # 3 * D
    
    # Reshape for kernel
    x_2d = x.view(B * T, D).contiguous()
    
    # Allocate output
    qkv = torch.empty((B * T, d_out), device=x.device, dtype=x.dtype)
    
    # Grid
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(B * T, BLOCK_M), triton.cdiv(d_out, BLOCK_N))
    
    fused_qkv_proj_kernel[grid](
        x_2d, weight, qkv,
        B * T, D, d_out,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(1), weight.stride(0),
        qkv.stride(0), qkv.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return qkv.view(B, T, d_out)


# =============================================================================
# ALIBI UTILITIES
# =============================================================================

def compute_alibi_slopes(n_heads: int, device: torch.device = None) -> torch.Tensor:
    """
    Compute ALiBi slopes for given number of heads.
    
    Following the original ALiBi paper, slopes are computed as:
    slopes[i] = 2^(-8 * i / n_heads) for i in [1, n_heads]
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = torch.arange(1, 1 + closest_power_of_2)
    slopes = torch.pow(base, powers)
    
    if closest_power_of_2 != n_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        num_rem = min(closest_power_of_2, n_heads - closest_power_of_2)
        extra_slopes = torch.pow(extra_base, torch.arange(1, 1 + 2 * num_rem, 2))
        slopes = torch.cat([slopes, extra_slopes])
    
    if device is not None:
        slopes = slopes.to(device)
    
    return slopes
