"""
Backward Pass Kernels for Algebraic Transformer

Triton kernels for attention backward pass with rational softmax.
Uses recomputation of attention weights (no storage) for memory efficiency.

Mathematical derivation for rational softmax gradients:

Forward:
    score_ij = (q_i @ k_j) * scale + alibi_bias_ij
    sigma_ij = 0.5 * (score_ij / (|score_ij| + 1) + 1)
    f_ij = sigma_ij^4  (unnormalized)
    L_i = Σ_j f_ij
    p_ij = f_ij / L_i  (normalized attention weights)
    o_i = Σ_j p_ij * v_j

Backward (given dL/do_i = do_i):
    
    dL/dv_j = Σ_i p_ij * do_i  (straightforward)
    
    For dL/dq_i and dL/dk_j, we need dL/dp_ij and dp_ij/dscore_ij.
    
    dL/dp_ij = do_i @ v_j^T  (element-wise for the (i,j) pair)
    
    dp_ij/dscore_kl:
        If i != k: 0
        If i = k, j = l: p_ij * g_ij * (1 - p_ij) where g_ij = 4 * σ'_ij / σ_ij
        If i = k, j != l: -p_ij * p_il * g_il
    
    So: dL/dscore_ij = Σ_l (dL/dp_il * dp_il/dscore_ij)
                     = dL/dp_ij * p_ij * g_ij - p_ij * Σ_l (dL/dp_il * p_il * g_il)
                     = p_ij * g_ij * (dL/dp_ij - Σ_l dL/dp_il * p_il * g_il / g_ij)
                     
    Simplified: dL/dscore_ij = p_ij * g_ij * (dL/dp_ij - D_i)
    where D_i = Σ_l (dL/dp_il * p_il * g_il) / avg(g)  [approximately]
    
    Actually, using standard softmax gradient pattern:
    ds_ij = p_ij * g_ij * (dL/dp_ij - Σ_l dL/dp_il * p_il)
          = p_ij * g_ij * (do_i @ v_j - Σ_l (do_i @ v_l) * p_il)
          = p_ij * g_ij * (do_i @ v_j - do_i @ o_i)
          = p_ij * g_ij * do_i @ (v_j - o_i)
          
    Then:
    dL/dq_i = Σ_j ds_ij * k_j * scale
    dL/dk_j = Σ_i ds_ij * q_i * scale
"""

import torch
import triton
import triton.language as tl
import math
from typing import Tuple

from .fused_ops import EPS, NEG_INF
from .forward import rational_attention_forward, compute_alibi_slopes


# =============================================================================
# BACKWARD KERNEL FOR dQ
# =============================================================================

@triton.jit
def rational_attention_bwd_dq_kernel(
    # Forward inputs
    Q_ptr, K_ptr, V_ptr,
    # Forward outputs
    Out_ptr, L_ptr,
    # Grad outputs
    dOut_ptr, dQ_ptr,
    # ALiBi
    alibi_slopes_ptr,
    # Dimensions
    seq_len, d_head, n_heads,
    # Strides
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_lb, stride_lh, stride_lt,
    # Parameters
    scale,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute gradient w.r.t. Q for one (batch, head, row)."""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    row_idx = tl.program_id(2)
    
    if row_idx >= seq_len:
        return
    
    alibi_slope = tl.load(alibi_slopes_ptr + head_idx)
    
    # Base pointers
    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    out_base = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    dout_base = dOut_ptr + batch_idx * stride_ob + head_idx * stride_oh
    dq_base = dQ_ptr + batch_idx * stride_qb + head_idx * stride_qh
    l_base = L_ptr + batch_idx * stride_lb + head_idx * stride_lh
    
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < d_head
    
    # Load row data
    q = tl.load(q_base + row_idx * stride_qt + d_offsets * stride_qd, 
                mask=d_mask, other=0.0).to(tl.float32)
    o = tl.load(out_base + row_idx * stride_ot + d_offsets * stride_od,
                mask=d_mask, other=0.0).to(tl.float32)
    do = tl.load(dout_base + row_idx * stride_ot + d_offsets * stride_od,
                 mask=d_mask, other=0.0).to(tl.float32)
    L_i = tl.load(l_base + row_idx * stride_lt).to(tl.float32)
    
    # Compute D_i = do @ o (for gradient correction)
    D_i = tl.sum(do * o)
    
    # Accumulate dQ
    dq = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    # Iterate over all valid key positions (causal)
    n_valid = row_idx + 1
    
    for col_idx in range(n_valid):
        # Load key and value
        k = tl.load(k_base + col_idx * stride_kt + d_offsets * stride_kd,
                   mask=d_mask, other=0.0).to(tl.float32)
        v = tl.load(v_base + col_idx * stride_vt + d_offsets * stride_vd,
                   mask=d_mask, other=0.0).to(tl.float32)
        
        # Recompute score
        score = tl.sum(q * k) * scale
        score = score - alibi_slope * tl.abs(row_idx - col_idx).to(tl.float32)
        
        # Recompute probability
        abs_score = tl.abs(score)
        denom = abs_score + 1.0
        sigma = 0.5 * (score / denom + 1.0)
        sigma2 = sigma * sigma
        f = sigma2 * sigma2  # σ^4
        p = f / (L_i + eps)
        
        # Compute g = 4 * σ' / σ = 2 / (denom^2 * σ)
        g = 2.0 / (denom * denom * (sigma + eps))
        
        # Compute dL/dp_ij = do @ v
        dl_dp = tl.sum(do * v)
        
        # Compute ds = p * g * (dL/dp - D_i)
        ds = p * g * (dl_dp - D_i)
        
        # Accumulate dQ contribution: ds * k * scale
        dq += ds * k * scale
    
    # Store dQ
    tl.store(dq_base + row_idx * stride_qt + d_offsets * stride_qd,
             dq.to(dQ_ptr.dtype.element_ty), mask=d_mask)


# =============================================================================
# BACKWARD KERNEL FOR dK, dV
# =============================================================================

@triton.jit
def rational_attention_bwd_dkv_kernel(
    # Forward inputs
    Q_ptr, K_ptr, V_ptr,
    # Forward outputs
    Out_ptr, L_ptr,
    # Grad outputs
    dOut_ptr, dK_ptr, dV_ptr,
    # ALiBi
    alibi_slopes_ptr,
    # Dimensions
    seq_len, d_head, n_heads,
    # Strides
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_lb, stride_lh, stride_lt,
    # Parameters
    scale,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute gradients w.r.t. K and V for one (batch, head, col)."""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    col_idx = tl.program_id(2)
    
    if col_idx >= seq_len:
        return
    
    alibi_slope = tl.load(alibi_slopes_ptr + head_idx)
    
    # Base pointers
    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    out_base = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    dout_base = dOut_ptr + batch_idx * stride_ob + head_idx * stride_oh
    dk_base = dK_ptr + batch_idx * stride_kb + head_idx * stride_kh
    dv_base = dV_ptr + batch_idx * stride_vb + head_idx * stride_vh
    l_base = L_ptr + batch_idx * stride_lb + head_idx * stride_lh
    
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < d_head
    
    # Load key and value for this column
    k = tl.load(k_base + col_idx * stride_kt + d_offsets * stride_kd,
               mask=d_mask, other=0.0).to(tl.float32)
    v = tl.load(v_base + col_idx * stride_vt + d_offsets * stride_vd,
               mask=d_mask, other=0.0).to(tl.float32)
    
    # Initialize accumulators
    dk = tl.zeros([BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    # Iterate over all query rows that attend to this key (causal: row >= col)
    for row_idx in range(col_idx, seq_len):
        # Load query, output, grad_output, L
        q = tl.load(q_base + row_idx * stride_qt + d_offsets * stride_qd,
                   mask=d_mask, other=0.0).to(tl.float32)
        o = tl.load(out_base + row_idx * stride_ot + d_offsets * stride_od,
                   mask=d_mask, other=0.0).to(tl.float32)
        do = tl.load(dout_base + row_idx * stride_ot + d_offsets * stride_od,
                    mask=d_mask, other=0.0).to(tl.float32)
        L_i = tl.load(l_base + row_idx * stride_lt).to(tl.float32)
        
        # Recompute score
        score = tl.sum(q * k) * scale
        score = score - alibi_slope * tl.abs(row_idx - col_idx).to(tl.float32)
        
        # Recompute probability
        abs_score = tl.abs(score)
        denom = abs_score + 1.0
        sigma = 0.5 * (score / denom + 1.0)
        sigma2 = sigma * sigma
        f = sigma2 * sigma2
        p = f / (L_i + eps)
        
        # Compute g = 4 * σ' / σ
        g = 2.0 / (denom * denom * (sigma + eps))
        
        # D_i = do @ o
        D_i = tl.sum(do * o)
        
        # dL/dp = do @ v
        dl_dp = tl.sum(do * v)
        
        # ds = p * g * (dL/dp - D_i)
        ds = p * g * (dl_dp - D_i)
        
        # dK contribution: ds * q * scale
        dk += ds * q * scale
        
        # dV contribution: p * do
        dv += p * do
    
    # Store gradients
    tl.store(dk_base + col_idx * stride_kt + d_offsets * stride_kd,
             dk.to(dK_ptr.dtype.element_ty), mask=d_mask)
    tl.store(dv_base + col_idx * stride_vt + d_offsets * stride_vd,
             dv.to(dV_ptr.dtype.element_ty), mask=d_mask)


# =============================================================================
# PYTHON INTERFACE
# =============================================================================

def rational_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    L: torch.Tensor,
    grad_output: torch.Tensor,
    alibi_slopes: torch.Tensor,
    scale: float,
    causal: bool = True,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for rational attention.
    """
    assert q.is_cuda
    
    B, H, T, D = q.shape
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = output.contiguous()
    L = L.contiguous()
    grad_output = grad_output.contiguous()
    alibi_slopes = alibi_slopes.contiguous()
    
    grad_q = torch.zeros_like(q)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)
    
    BLOCK_D = triton.next_power_of_2(D)
    
    # Compute dQ
    grid = (B, H, T)
    rational_attention_bwd_dq_kernel[grid](
        q, k, v,
        output, L,
        grad_output, grad_q,
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
    )
    
    # Compute dK, dV
    rational_attention_bwd_dkv_kernel[grid](
        q, k, v,
        output, L,
        grad_output, grad_k, grad_v,
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
    )
    
    return grad_q, grad_k, grad_v


# =============================================================================
# COMBINED AUTOGRAD FUNCTION
# =============================================================================

class RationalAttentionFunction(torch.autograd.Function):
    """Complete autograd function for rational attention."""
    
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
        output, L = rational_attention_forward(q, k, v, alibi_slopes, scale, causal, eps)
        
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
    """Memory-efficient rational attention with custom backward."""
    return RationalAttentionFunction.apply(q, k, v, alibi_slopes, scale, causal, eps)
