"""
Fused Operations for Algebraic Transformer

Core Triton kernels and utilities used by both forward and backward passes.
All operations are purely algebraic (no transcendentals like exp, log, sqrt).

Key operations:
- Rational sigmoid: σ(x) = 0.5 * (x / (|x| + 1) + 1)
- Rational softmax: p_i = (σ(x_i))^4 / Σ_j (σ(x_j))^4
- Rational SwiGLU: gate * σ(gate) * value
- Mean-error normalization: x / (mean(|x|) + ε) * γ
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


# =============================================================================
# CONFIGURATION AND UTILITIES
# =============================================================================

# Numerical stability constants
EPS = 1e-6
NEG_INF = -1e4  # Used for masking (not -inf to keep gradients stable)

# Autotuning configurations
SOFTMAX_CONFIGS = [
    triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
]

NORM_CONFIGS = [
    triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
]


def get_optimal_block_size(n: int, max_block: int = 2048) -> int:
    """Get optimal block size for a given dimension."""
    return min(triton.next_power_of_2(n), max_block)


# =============================================================================
# RATIONAL FUNCTION PRIMITIVES (used in multiple kernels)
# =============================================================================

@triton.jit
def rational_sigmoid(x):
    """
    Rational sigmoid approximation: σ(x) = 0.5 * (x / (|x| + 1) + 1)
    
    Properties:
    - Range: (0, 1)
    - σ(0) = 0.5
    - Monotonically increasing
    - Derivative: dσ/dx = 0.5 / (|x| + 1)^2
    """
    abs_x = tl.abs(x)
    return 0.5 * (x / (abs_x + 1.0) + 1.0)


@triton.jit
def rational_sigmoid_grad(x):
    """
    Derivative of rational sigmoid: dσ/dx = 0.5 / (|x| + 1)^2
    """
    abs_x = tl.abs(x)
    denom = abs_x + 1.0
    return 0.5 / (denom * denom)


@triton.jit  
def rational_softmax_prob(x):
    """
    Compute unnormalized rational softmax probability: p = σ(x)^4
    
    Uses σ(x)^4 because:
    - Higher power creates sharper attention distributions
    - Still bounded in [0, 1] before normalization
    - Gradient is well-behaved
    """
    s = rational_sigmoid(x)
    s2 = s * s
    return s2 * s2  # s^4


@triton.jit
def rational_softmax_prob_and_grad(x):
    """
    Compute both σ(x)^4 and its gradient d(σ^4)/dx = 4σ^3 * dσ/dx
    
    Returns: (prob, d_prob_dx)
    """
    abs_x = tl.abs(x)
    denom = abs_x + 1.0
    
    # σ(x) = 0.5 * (x/denom + 1)
    s = 0.5 * (x / denom + 1.0)
    
    # dσ/dx = 0.5 / denom^2
    ds_dx = 0.5 / (denom * denom)
    
    # p = s^4
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    
    # dp/dx = 4 * s^3 * ds/dx
    dp_dx = 4.0 * s3 * ds_dx
    
    return s4, dp_dx


# =============================================================================
# RATIONAL SOFTMAX KERNELS
# =============================================================================

@triton.autotune(configs=SOFTMAX_CONFIGS, key=['N'])
@triton.jit
def rational_softmax_fwd_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    stride_input,
    stride_output,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward kernel for rational softmax over the last dimension.
    
    For each row, computes:
        p_i = σ(x_i)^4 / Σ_j σ(x_j)^4
    
    Uses two-pass algorithm:
    1. First pass: compute sum of σ(x)^4
    2. Second pass: normalize and store
    """
    row_idx = tl.program_id(0)
    
    input_row_ptr = input_ptr + row_idx * stride_input
    output_row_ptr = output_ptr + row_idx * stride_output
    
    # First pass: accumulate sum
    total_sum = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        
        x = tl.load(input_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        p = rational_softmax_prob(x)
        total_sum += tl.sum(tl.where(mask, p, 0.0))
    
    # Second pass: normalize and store
    inv_sum = 1.0 / (total_sum + eps)
    
    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        
        x = tl.load(input_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        p = rational_softmax_prob(x)
        normalized_p = p * inv_sum
        
        tl.store(output_row_ptr + col_offsets, normalized_p.to(output_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=SOFTMAX_CONFIGS, key=['N'])
@triton.jit
def rational_softmax_bwd_kernel(
    grad_output_ptr,
    output_ptr,  # Forward output (normalized probs)
    input_ptr,   # Forward input (scores)
    grad_input_ptr,
    N: tl.constexpr,
    stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward kernel for rational softmax.
    
    Given: p_i = s_i^4 / Z, where s_i = σ(x_i) and Z = Σ s_j^4
    
    Gradient: dp_i/dx_j = (δ_ij * 4s_i^3 * σ'(x_i) * Z - s_i^4 * 4s_j^3 * σ'(x_j)) / Z^2
                        = p_i * (δ_ij * 4σ'(x_i)/s_i - 4s_j^3 * σ'(x_j) / Z)
    
    For efficiency, we compute:
    dp_i/dx_i = p_i * 4σ'(x_i) * (1/s_i - p_i * s_i^2/s_i^4) 
              = p_i * 4σ'(x_i) * (1 - p_i) / s_i  [approximately, simplified]
    
    Actually using the standard softmax-style gradient formula adapted:
    dL/dx_i = Σ_j (dL/dp_j * dp_j/dx_i)
            = dL/dp_i * p_i * local_grad_i - p_i * Σ_j(dL/dp_j * p_j * local_grad_j)
    
    Where local_grad_i = d(log p_i)/dx_i before normalization = 4σ'(x_i)/σ(x_i)
    """
    row_idx = tl.program_id(0)
    
    grad_out_row = grad_output_ptr + row_idx * stride
    output_row = output_ptr + row_idx * stride
    input_row = input_ptr + row_idx * stride
    grad_in_row = grad_input_ptr + row_idx * stride
    
    # First pass: compute weighted sum for gradient correction
    # sum_term = Σ_j (grad_out_j * p_j * local_grad_j)
    sum_term = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        
        grad_out = tl.load(grad_out_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(output_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(input_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Compute local gradient factor: 4 * σ'(x) / σ(x)
        # σ(x) = 0.5 * (x/(|x|+1) + 1)
        # σ'(x) = 0.5 / (|x|+1)^2
        abs_x = tl.abs(x)
        denom = abs_x + 1.0
        sigma = 0.5 * (x / denom + 1.0)
        sigma_grad = 0.5 / (denom * denom)
        
        # Avoid division by zero when sigma is very small
        local_grad = 4.0 * sigma_grad / (sigma + 1e-8)
        
        sum_term += tl.sum(tl.where(mask, grad_out * p * local_grad, 0.0))
    
    # Second pass: compute and store gradients
    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        
        grad_out = tl.load(grad_out_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(output_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(input_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Recompute local gradient
        abs_x = tl.abs(x)
        denom = abs_x + 1.0
        sigma = 0.5 * (x / denom + 1.0)
        sigma_grad = 0.5 / (denom * denom)
        local_grad = 4.0 * sigma_grad / (sigma + 1e-8)
        
        # Gradient formula: grad_in = p * local_grad * (grad_out - sum_term)
        grad_in = p * local_grad * (grad_out - sum_term)
        
        tl.store(grad_in_row + col_offsets, grad_in.to(grad_input_ptr.dtype.element_ty), mask=mask)


# =============================================================================
# MEAN-ERROR NORMALIZATION KERNELS
# =============================================================================

@triton.autotune(configs=NORM_CONFIGS, key=['N'])
@triton.jit
def mean_error_norm_fwd_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N: tl.constexpr,
    stride_input,
    stride_output,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward kernel for mean-error normalization.
    
    output = (x / (mean(|x|) + eps)) * weight
    """
    row_idx = tl.program_id(0)
    
    input_row = input_ptr + row_idx * stride_input
    output_row = output_ptr + row_idx * stride_output
    
    # First pass: compute mean of absolute values
    abs_sum = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        
        x = tl.load(input_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        abs_sum += tl.sum(tl.where(mask, tl.abs(x), 0.0))
    
    mean_abs = abs_sum / N
    inv_mean = 1.0 / (mean_abs + eps)
    
    # Second pass: normalize and apply weight
    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        
        x = tl.load(input_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        normalized = x * inv_mean * w
        
        tl.store(output_row + col_offsets, normalized.to(output_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=NORM_CONFIGS, key=['N'])
@triton.jit
def mean_error_norm_bwd_kernel(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    grad_input_ptr,
    grad_weight_ptr,  # Accumulated atomically
    N: tl.constexpr,
    stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward kernel for mean-error normalization.
    
    Forward: y = (x / μ) * w, where μ = mean(|x|) + eps
    
    Gradients:
    - dy/dx_i = w_i/μ - (x_i * sign(x_i) / (N * μ^2)) * Σ_j (w_j * x_j / μ)
              = w_i/μ - (sign(x_i) / (N * μ)) * Σ_j (y_j * grad_out_j) * (something)
    
    Simplified gradient computation using chain rule.
    """
    row_idx = tl.program_id(0)
    
    grad_out_row = grad_output_ptr + row_idx * stride
    input_row = input_ptr + row_idx * stride
    grad_in_row = grad_input_ptr + row_idx * stride
    
    # First pass: compute mean(|x|) and dot product of grad_out with normalized x
    abs_sum = tl.zeros([1], dtype=tl.float32)
    dot_prod = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        
        x = tl.load(input_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        abs_x = tl.abs(x)
        abs_sum += tl.sum(tl.where(mask, abs_x, 0.0))
        
        # grad_out * w * x / μ contributes to correction term
        dot_prod += tl.sum(tl.where(mask, grad_out * w * x, 0.0))
    
    mean_abs = abs_sum / N
    inv_mean = 1.0 / (mean_abs + eps)
    
    # Correction factor for gradient
    correction = dot_prod * inv_mean * inv_mean / N
    
    # Second pass: compute gradients
    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        
        x = tl.load(input_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Sign of x for gradient correction
        sign_x = tl.where(x >= 0, 1.0, -1.0)
        
        # Gradient w.r.t input
        grad_in = grad_out * w * inv_mean - sign_x * correction
        
        tl.store(grad_in_row + col_offsets, grad_in.to(grad_input_ptr.dtype.element_ty), mask=mask)
        
        # Gradient w.r.t weight (accumulated across rows)
        grad_w = grad_out * x * inv_mean
        tl.atomic_add(grad_weight_ptr + col_offsets, grad_w, mask=mask)


# =============================================================================
# SWIGLU KERNELS
# =============================================================================

@triton.jit
def swiglu_fwd_kernel(
    gate_ptr,
    value_ptr,
    output_ptr,
    N: tl.constexpr,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward kernel for rational SwiGLU.
    
    output = gate * σ(gate) * value
    
    Where σ is the rational sigmoid.
    """
    row_idx = tl.program_id(0)
    
    gate_row = gate_ptr + row_idx * stride
    value_row = value_ptr + row_idx * stride
    output_row = output_ptr + row_idx * stride
    
    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        
        gate = tl.load(gate_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(value_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # SwiGLU: gate * σ(gate) * value
        sigma_gate = rational_sigmoid(gate)
        output = gate * sigma_gate * value
        
        tl.store(output_row + col_offsets, output.to(output_ptr.dtype.element_ty), mask=mask)


@triton.jit
def swiglu_bwd_kernel(
    grad_output_ptr,
    gate_ptr,
    value_ptr,
    grad_gate_ptr,
    grad_value_ptr,
    N: tl.constexpr,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward kernel for rational SwiGLU.
    
    Forward: y = gate * σ(gate) * value
    
    Gradients:
    - dy/dgate = value * (σ(gate) + gate * σ'(gate))
    - dy/dvalue = gate * σ(gate)
    """
    row_idx = tl.program_id(0)
    
    grad_out_row = grad_output_ptr + row_idx * stride
    gate_row = gate_ptr + row_idx * stride
    value_row = value_ptr + row_idx * stride
    grad_gate_row = grad_gate_ptr + row_idx * stride
    grad_value_row = grad_value_ptr + row_idx * stride
    
    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        
        grad_out = tl.load(grad_out_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(value_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Compute sigmoid and its derivative
        sigma = rational_sigmoid(gate)
        sigma_grad = rational_sigmoid_grad(gate)
        
        # Gradients
        grad_gate = grad_out * value * (sigma + gate * sigma_grad)
        grad_value = grad_out * gate * sigma
        
        tl.store(grad_gate_row + col_offsets, grad_gate.to(grad_gate_ptr.dtype.element_ty), mask=mask)
        tl.store(grad_value_row + col_offsets, grad_value.to(grad_value_ptr.dtype.element_ty), mask=mask)


# =============================================================================
# PYTHON WRAPPER FUNCTIONS
# =============================================================================

class RationalSoftmax(torch.autograd.Function):
    """Autograd function for rational softmax with custom CUDA kernels."""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, eps: float = EPS) -> torch.Tensor:
        assert input.is_cuda, "Input must be on CUDA"
        
        # Reshape to 2D for kernel
        original_shape = input.shape
        input_2d = input.contiguous().view(-1, input.shape[-1])
        n_rows, n_cols = input_2d.shape
        
        # Allocate output
        output = torch.empty_like(input_2d)
        
        # Launch kernel
        grid = (n_rows,)
        rational_softmax_fwd_kernel[grid](
            input_2d, output,
            n_cols,
            input_2d.stride(0), output.stride(0),
            eps,
        )
        
        # Save for backward
        output_reshaped = output.view(original_shape)
        ctx.save_for_backward(input, output_reshaped)
        ctx.eps = eps
        
        return output_reshaped
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, output = ctx.saved_tensors
        eps = ctx.eps
        
        # Reshape to 2D
        original_shape = input.shape
        input_2d = input.contiguous().view(-1, input.shape[-1])
        output_2d = output.contiguous().view(-1, output.shape[-1])
        grad_output_2d = grad_output.contiguous().view(-1, grad_output.shape[-1])
        n_rows, n_cols = input_2d.shape
        
        # Allocate gradient
        grad_input = torch.empty_like(input_2d)
        
        # Launch kernel
        grid = (n_rows,)
        rational_softmax_bwd_kernel[grid](
            grad_output_2d, output_2d, input_2d, grad_input,
            n_cols,
            input_2d.stride(0),
            eps,
        )
        
        return grad_input.view(original_shape), None


class MeanErrorNorm(torch.autograd.Function):
    """Autograd function for mean-error normalization."""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
        assert input.is_cuda, "Input must be on CUDA"
        
        # Reshape to 2D
        original_shape = input.shape
        input_2d = input.contiguous().view(-1, input.shape[-1])
        n_rows, n_cols = input_2d.shape
        
        # Allocate output
        output = torch.empty_like(input_2d)
        
        # Launch kernel
        grid = (n_rows,)
        mean_error_norm_fwd_kernel[grid](
            input_2d, weight, output,
            n_cols,
            input_2d.stride(0), output.stride(0),
            eps,
        )
        
        ctx.save_for_backward(input, weight)
        ctx.eps = eps
        
        return output.view(original_shape)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors
        eps = ctx.eps
        
        # Reshape to 2D
        original_shape = input.shape
        input_2d = input.contiguous().view(-1, input.shape[-1])
        grad_output_2d = grad_output.contiguous().view(-1, grad_output.shape[-1])
        n_rows, n_cols = input_2d.shape
        
        # Allocate gradients
        grad_input = torch.empty_like(input_2d)
        grad_weight = torch.zeros_like(weight)
        
        # Launch kernel
        grid = (n_rows,)
        mean_error_norm_bwd_kernel[grid](
            grad_output_2d, input_2d, weight, grad_input, grad_weight,
            n_cols,
            input_2d.stride(0),
            eps,
        )
        
        return grad_input.view(original_shape), grad_weight, None


class RationalSwiGLU(torch.autograd.Function):
    """Autograd function for rational SwiGLU."""
    
    @staticmethod
    def forward(ctx, gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        assert gate.is_cuda and value.is_cuda, "Inputs must be on CUDA"
        
        # Reshape to 2D
        original_shape = gate.shape
        gate_2d = gate.contiguous().view(-1, gate.shape[-1])
        value_2d = value.contiguous().view(-1, value.shape[-1])
        n_rows, n_cols = gate_2d.shape
        
        # Allocate output
        output = torch.empty_like(gate_2d)
        
        # Choose block size
        BLOCK_SIZE = get_optimal_block_size(n_cols)
        
        # Launch kernel
        grid = (n_rows,)
        swiglu_fwd_kernel[grid](
            gate_2d, value_2d, output,
            n_cols,
            gate_2d.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        ctx.save_for_backward(gate, value)
        
        return output.view(original_shape)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, value = ctx.saved_tensors
        
        # Reshape to 2D
        original_shape = gate.shape
        gate_2d = gate.contiguous().view(-1, gate.shape[-1])
        value_2d = value.contiguous().view(-1, value.shape[-1])
        grad_output_2d = grad_output.contiguous().view(-1, grad_output.shape[-1])
        n_rows, n_cols = gate_2d.shape
        
        # Allocate gradients
        grad_gate = torch.empty_like(gate_2d)
        grad_value = torch.empty_like(value_2d)
        
        # Choose block size
        BLOCK_SIZE = get_optimal_block_size(n_cols)
        
        # Launch kernel
        grid = (n_rows,)
        swiglu_bwd_kernel[grid](
            grad_output_2d, gate_2d, value_2d, grad_gate, grad_value,
            n_cols,
            gate_2d.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return grad_gate.view(original_shape), grad_value.view(original_shape)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def rational_softmax(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Apply rational softmax to last dimension."""
    if x.is_cuda:
        return RationalSoftmax.apply(x, eps)
    else:
        # CPU fallback
        abs_x = x.abs()
        s = x / (abs_x + 1.0)
        p_base = (s + 1.0) * 0.5
        p4 = p_base.pow(4)
        return p4 / (p4.sum(dim=-1, keepdim=True) + eps)


def mean_error_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Apply mean-error normalization."""
    if x.is_cuda:
        return MeanErrorNorm.apply(x, weight, eps)
    else:
        # CPU fallback
        magnitude = x.abs().mean(dim=-1, keepdim=True)
        return (x / (magnitude + eps)) * weight


def rational_swiglu(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply rational SwiGLU activation."""
    if gate.is_cuda:
        return RationalSwiGLU.apply(gate, value)
    else:
        # CPU fallback
        abs_gate = gate.abs()
        sigmoid_approx = 0.5 * (gate / (abs_gate + 1.0) + 1.0)
        return gate * sigmoid_approx * value
