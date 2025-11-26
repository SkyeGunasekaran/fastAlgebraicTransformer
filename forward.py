"""
Forward Pass Kernels for Algebraic Transformer

Optimized Triton kernels for attention forward pass with rational softmax.
Uses a simple, correct implementation prioritizing numerical accuracy.
"""

import torch
import triton
import triton.language as tl
import math
from typing import Tuple

from .fused_ops import EPS, NEG_INF


# =============================================================================
# ALIBI UTILITIES
# =============================================================================

def compute_alibi_slopes(n_heads: int, device: torch.device = None) -> torch.Tensor:
    """
    Compute ALiBi slopes for given number of heads.
    
    Following the original ALiBi paper:
    slopes[i] = 2^(-8 * (i+1) / n_heads) for power-of-2 heads
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
    
    return slopes.float()


# =============================================================================
# SIMPLE ATTENTION FORWARD (Non-tiled, for correctness)
# =============================================================================

@triton.jit
def rational_attention_fwd_kernel(
    # Input pointers
    Q_ptr, K_ptr, V_ptr,
    # Output pointers
    Out_ptr, L_ptr,
    # ALiBi slopes
    alibi_slopes_ptr,
    # Dimensions  
    seq_len, d_head,
    n_heads,
    # Strides for Q, K, V, Out: [B, H, T, D]
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    # Strides for L: [B, H, T]
    stride_lb, stride_lh, stride_lt,
    # Parameters
    scale,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Simple attention forward kernel.
    
    Each program handles one (batch, head, query_row).
    Iterates over all key positions to compute attention for that row.
    """
    # Program indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    row_idx = tl.program_id(2)
    
    # Early exit if row is out of bounds
    if row_idx >= seq_len:
        return
    
    # Load ALiBi slope for this head
    alibi_slope = tl.load(alibi_slopes_ptr + head_idx)
    
    # Compute base pointers for this (batch, head)
    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    out_base = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    l_base = L_ptr + batch_idx * stride_lb + head_idx * stride_lh
    
    # Load query vector for this row
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < d_head
    
    q_ptr = q_base + row_idx * stride_qt
    q = tl.load(q_ptr + d_offsets * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    l_sum = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over key positions (causal: only up to row_idx)
    for col_start in range(0, row_idx + 1, BLOCK_N):
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        col_mask = (col_offsets <= row_idx) & (col_offsets < seq_len)
        
        # Compute scores for this block of keys
        for i in range(BLOCK_N):
            col_idx = col_start + i
            
            # Skip if out of causal range
            if col_idx > row_idx:
                continue
            if col_idx >= seq_len:
                continue
            
            # Load key vector
            k_ptr = k_base + col_idx * stride_kt
            k = tl.load(k_ptr + d_offsets * stride_kd, mask=d_mask, other=0.0).to(tl.float32)
            
            # Compute attention score: q @ k * scale
            score = tl.sum(q * k) * scale
            
            # Add ALiBi bias: -slope * |row - col|
            distance = tl.abs(row_idx - col_idx)
            score = score - alibi_slope * distance.to(tl.float32)
            
            # Compute rational softmax probability (unnormalized)
            # p = σ(score)^4 where σ(x) = 0.5 * (x / (|x| + 1) + 1)
            abs_score = tl.abs(score)
            sigma = 0.5 * (score / (abs_score + 1.0) + 1.0)
            sigma2 = sigma * sigma
            prob = sigma2 * sigma2  # σ^4
            
            l_sum += prob
            
            # Load value vector and accumulate
            v_ptr = v_base + col_idx * stride_vt
            v = tl.load(v_ptr + d_offsets * stride_vd, mask=d_mask, other=0.0).to(tl.float32)
            acc += prob * v
    
    # Normalize
    inv_l = 1.0 / (l_sum + eps)
    acc = acc * inv_l
    
    # Store output
    out_ptr = out_base + row_idx * stride_ot
    tl.store(out_ptr + d_offsets * stride_od, acc.to(Out_ptr.dtype.element_ty), mask=d_mask)
    
    # Store normalization factor for backward pass
    l_ptr = l_base + row_idx * stride_lt
    tl.store(l_ptr, l_sum)


# =============================================================================
# VECTORIZED ATTENTION FORWARD (Better performance)
# =============================================================================

@triton.jit  
def rational_attention_fwd_vectorized_kernel(
    # Input pointers
    Q_ptr, K_ptr, V_ptr,
    # Output pointers
    Out_ptr, L_ptr,
    # ALiBi slopes
    alibi_slopes_ptr,
    # Dimensions
    seq_len, d_head,
    n_heads,
    # Strides for Q, K, V, Out: [B, H, T, D]
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    # Strides for L: [B, H, T]
    stride_lb, stride_lh, stride_lt,
    # Parameters
    scale,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Vectorized attention forward - processes BLOCK_N keys at a time.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    row_idx = tl.program_id(2)
    
    if row_idx >= seq_len:
        return
    
    alibi_slope = tl.load(alibi_slopes_ptr + head_idx)
    
    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    out_base = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    l_base = L_ptr + batch_idx * stride_lb + head_idx * stride_lh
    
    # Load query
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < d_head
    
    q = tl.load(q_base + row_idx * stride_qt + d_offsets * stride_qd, 
                mask=d_mask, other=0.0).to(tl.float32)
    
    # Initialize accumulators  
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    l_sum = tl.zeros([1], dtype=tl.float32)
    
    # Number of valid key positions (causal)
    n_valid = row_idx + 1
    
    # Iterate over key blocks
    for col_start in range(0, n_valid, BLOCK_N):
        n_offsets = tl.arange(0, BLOCK_N)
        col_indices = col_start + n_offsets
        col_mask = col_indices < n_valid
        
        # Compute scores for this block
        scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        for i in range(BLOCK_N):
            col_idx = col_start + i
            if col_idx < n_valid:
                k = tl.load(k_base + col_idx * stride_kt + d_offsets * stride_kd,
                           mask=d_mask, other=0.0).to(tl.float32)
                s = tl.sum(q * k) * scale
                s = s - alibi_slope * tl.abs(row_idx - col_idx).to(tl.float32)
                # Store in scores array
                # Note: We compute one at a time since tl.where on arrays is tricky
                
                # Compute probability
                abs_s = tl.abs(s)
                sigma = 0.5 * (s / (abs_s + 1.0) + 1.0)
                sigma2 = sigma * sigma
                prob = sigma2 * sigma2
                
                l_sum += prob
                
                # Load value and accumulate
                v = tl.load(v_base + col_idx * stride_vt + d_offsets * stride_vd,
                           mask=d_mask, other=0.0).to(tl.float32)
                acc += prob * v
    
    # Normalize
    acc = acc / (l_sum + eps)
    
    # Store
    tl.store(out_base + row_idx * stride_ot + d_offsets * stride_od,
             acc.to(Out_ptr.dtype.element_ty), mask=d_mask)
    tl.store(l_base + row_idx * stride_lt, l_sum)


# =============================================================================
# PYTHON INTERFACE
# =============================================================================

def rational_attention_forward(
    q: torch.Tensor,  # [B, H, T, D]
    k: torch.Tensor,
    v: torch.Tensor,
    alibi_slopes: torch.Tensor,  # [H]
    scale: float,
    causal: bool = True,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for rational attention.
    
    Returns:
        output: [B, H, T, D]
        L: [B, H, T] normalization factors for backward
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape
    
    B, H, T, D = q.shape
    
    q = q.contiguous()
    k = k.contiguous()  
    v = v.contiguous()
    alibi_slopes = alibi_slopes.contiguous()
    
    output = torch.empty_like(q)
    L = torch.empty((B, H, T), device=q.device, dtype=torch.float32)
    
    # Choose block sizes
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_N = 16  # Process keys one at a time for correctness
    
    # Grid: one program per (batch, head, query_row)
    grid = (B, H, T)
    
    rational_attention_fwd_kernel[grid](
        q, k, v,
        output, L,
        alibi_slopes,
        T, D, H,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        scale,
        eps,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
    )
    
    return output, L
