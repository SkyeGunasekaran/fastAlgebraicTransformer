"""
Fused Operations for Algebraic Transformer

Core Triton kernels for purely algebraic (rational) operations.
All operations avoid transcendental functions (exp, log, sqrt).

Key operations:
- Rational sigmoid: σ(x) = 0.5 * (x / (|x| + 1) + 1)
- Rational softmax: p_i = σ(x_i)^4 / Σ_j σ(x_j)^4
- Rational SwiGLU: gate * σ(gate) * value  
- Mean-error normalization: x / (mean(|x|) + ε) * γ
"""

import torch
import triton
import triton.language as tl
from typing import Optional

# Numerical stability constant
EPS = 1e-6
NEG_INF = -1e4


# =============================================================================
# TRITON KERNEL PRIMITIVES
# =============================================================================

@triton.jit
def rational_sigmoid(x):
    """
    Rational sigmoid: σ(x) = 0.5 * (x / (|x| + 1) + 1)
    Range: (0, 1), σ(0) = 0.5
    """
    abs_x = tl.abs(x)
    return 0.5 * (x / (abs_x + 1.0) + 1.0)


@triton.jit
def rational_sigmoid_deriv(x):
    """
    Derivative of rational sigmoid: dσ/dx = 0.5 / (|x| + 1)^2
    """
    abs_x = tl.abs(x)
    denom = abs_x + 1.0
    return 0.5 / (denom * denom)


# =============================================================================
# RATIONAL SOFTMAX FORWARD KERNEL
# =============================================================================

@triton.jit
def rational_softmax_fwd_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward kernel for rational softmax.
    p_i = σ(x_i)^4 / Σ_j σ(x_j)^4
    """
    row_idx = tl.program_id(0)
    
    row_start_in = input_ptr + row_idx * input_row_stride
    row_start_out = output_ptr + row_idx * output_row_stride
    
    # First pass: compute sum of σ(x)^4
    total = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x = tl.load(row_start_in + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # σ(x) = 0.5 * (x / (|x| + 1) + 1)
        abs_x = tl.abs(x)
        sigma = 0.5 * (x / (abs_x + 1.0) + 1.0)
        
        # σ^4
        sigma2 = sigma * sigma
        sigma4 = sigma2 * sigma2
        
        total += tl.sum(tl.where(mask, sigma4, 0.0))
    
    inv_total = 1.0 / (total + eps)
    
    # Second pass: normalize and store
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x = tl.load(row_start_in + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        abs_x = tl.abs(x)
        sigma = 0.5 * (x / (abs_x + 1.0) + 1.0)
        sigma2 = sigma * sigma
        sigma4 = sigma2 * sigma2
        
        p = sigma4 * inv_total
        
        tl.store(row_start_out + col_offsets, p.to(output_ptr.dtype.element_ty), mask=mask)


# =============================================================================
# RATIONAL SOFTMAX BACKWARD KERNEL
# =============================================================================

@triton.jit
def rational_softmax_bwd_kernel(
    grad_output_ptr,
    input_ptr,
    output_ptr,
    grad_input_ptr,
    n_cols,
    stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward kernel for rational softmax.
    
    Forward: p_i = σ_i^4 / Z, where Z = Σ_j σ_j^4
    
    Let f_i = σ_i^4 (unnormalized), so p_i = f_i / Z
    
    df_i/dx_i = 4 * σ_i^3 * σ'_i = 4 * σ_i^3 * (0.5 / (|x_i| + 1)^2)
              = 2 * σ_i^3 / (|x_i| + 1)^2
    
    dp_i/dx_i = (df_i/dx_i * Z - f_i * df_i/dx_i) / Z^2
              = (df_i/dx_i / Z) * (1 - p_i)
              = p_i * (df_i/dx_i / f_i) * (1 - p_i)
              = p_i * (4 * σ'_i / σ_i) * (1 - p_i)
    
    dp_i/dx_j (j ≠ i) = -f_i * df_j/dx_j / Z^2
                      = -p_i * p_j * (4 * σ'_j / σ_j)
    
    dL/dx_i = Σ_j (dL/dp_j * dp_j/dx_i)
            = dL/dp_i * p_i * g_i * (1 - p_i) - Σ_{j≠i} dL/dp_j * p_j * p_i * g_i
            = p_i * g_i * (dL/dp_i * (1 - p_i) - Σ_{j≠i} dL/dp_j * p_j)
            = p_i * g_i * (dL/dp_i - Σ_j dL/dp_j * p_j)
            = p_i * g_i * (dL/dp_i - dot(dL/dp, p))
    
    where g_i = 4 * σ'_i / σ_i = 2 / ((|x_i| + 1)^2 * σ_i)
    """
    row_idx = tl.program_id(0)
    
    grad_out_row = grad_output_ptr + row_idx * stride
    input_row = input_ptr + row_idx * stride
    output_row = output_ptr + row_idx * stride
    grad_in_row = grad_input_ptr + row_idx * stride
    
    # First pass: compute dot(grad_out, p)
    dot_grad_p = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        grad_out = tl.load(grad_out_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(output_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        dot_grad_p += tl.sum(tl.where(mask, grad_out * p, 0.0))
    
    # Second pass: compute gradients
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x = tl.load(input_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(output_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Compute g = 4 * σ' / σ = 2 / ((|x| + 1)^2 * σ)
        abs_x = tl.abs(x)
        denom = abs_x + 1.0
        sigma = 0.5 * (x / denom + 1.0)
        
        # g = 4 * (0.5 / denom^2) / σ = 2 / (denom^2 * σ)
        g = 2.0 / (denom * denom * (sigma + eps))
        
        # grad_in = p * g * (grad_out - dot_grad_p)
        grad_in = p * g * (grad_out - dot_grad_p)
        
        tl.store(grad_in_row + col_offsets, grad_in.to(grad_input_ptr.dtype.element_ty), mask=mask)


# =============================================================================
# SWIGLU FORWARD KERNEL
# =============================================================================

@triton.jit
def swiglu_fwd_kernel(
    gate_ptr,
    value_ptr,
    output_ptr,
    n_cols,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward kernel for rational SwiGLU.
    output = gate * σ(gate) * value
    """
    row_idx = tl.program_id(0)
    
    gate_row = gate_ptr + row_idx * stride
    value_row = value_ptr + row_idx * stride
    output_row = output_ptr + row_idx * stride
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        gate = tl.load(gate_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(value_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # σ(gate)
        abs_gate = tl.abs(gate)
        sigma = 0.5 * (gate / (abs_gate + 1.0) + 1.0)
        
        # output = gate * σ(gate) * value
        out = gate * sigma * value
        
        tl.store(output_row + col_offsets, out.to(output_ptr.dtype.element_ty), mask=mask)


# =============================================================================
# SWIGLU BACKWARD KERNEL
# =============================================================================

@triton.jit
def swiglu_bwd_kernel(
    grad_output_ptr,
    gate_ptr,
    value_ptr,
    grad_gate_ptr,
    grad_value_ptr,
    n_cols,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward kernel for rational SwiGLU.
    
    Forward: y = gate * σ(gate) * value
    
    dy/dvalue = gate * σ(gate)
    
    dy/dgate = value * d(gate * σ(gate))/dgate
             = value * (σ(gate) + gate * σ'(gate))
    
    where σ'(gate) = 0.5 / (|gate| + 1)^2
    """
    row_idx = tl.program_id(0)
    
    grad_out_row = grad_output_ptr + row_idx * stride
    gate_row = gate_ptr + row_idx * stride
    value_row = value_ptr + row_idx * stride
    grad_gate_row = grad_gate_ptr + row_idx * stride
    grad_value_row = grad_value_ptr + row_idx * stride
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        grad_out = tl.load(grad_out_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(value_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Compute σ(gate) and σ'(gate)
        abs_gate = tl.abs(gate)
        denom = abs_gate + 1.0
        sigma = 0.5 * (gate / denom + 1.0)
        sigma_deriv = 0.5 / (denom * denom)
        
        # dy/dvalue = gate * σ(gate)
        grad_value = grad_out * gate * sigma
        
        # dy/dgate = value * (σ(gate) + gate * σ'(gate))
        grad_gate = grad_out * value * (sigma + gate * sigma_deriv)
        
        tl.store(grad_gate_row + col_offsets, grad_gate.to(grad_gate_ptr.dtype.element_ty), mask=mask)
        tl.store(grad_value_row + col_offsets, grad_value.to(grad_value_ptr.dtype.element_ty), mask=mask)


# =============================================================================
# MEAN-ERROR NORMALIZATION FORWARD KERNEL
# =============================================================================

@triton.jit
def mean_error_norm_fwd_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    mean_ptr,  # Store mean for backward
    n_cols,
    input_stride,
    output_stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward kernel for mean-error normalization.
    output = (x / (mean(|x|) + eps)) * weight
    """
    row_idx = tl.program_id(0)
    
    input_row = input_ptr + row_idx * input_stride
    output_row = output_ptr + row_idx * output_stride
    
    # First pass: compute mean(|x|)
    abs_sum = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x = tl.load(input_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        abs_sum += tl.sum(tl.where(mask, tl.abs(x), 0.0))
    
    mean_abs = abs_sum / n_cols
    inv_mean = 1.0 / (mean_abs + eps)
    
    # Store mean for backward pass
    tl.store(mean_ptr + row_idx, mean_abs)
    
    # Second pass: normalize and scale
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x = tl.load(input_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        
        out = x * inv_mean * w
        
        tl.store(output_row + col_offsets, out.to(output_ptr.dtype.element_ty), mask=mask)


# =============================================================================
# MEAN-ERROR NORMALIZATION BACKWARD KERNEL
# =============================================================================

@triton.jit
def mean_error_norm_bwd_kernel(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    mean_ptr,
    grad_input_ptr,
    n_cols,
    stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward kernel for mean-error normalization (input gradient only).
    
    Forward: y_i = (x_i / μ) * w_i, where μ = mean(|x|) + eps
    
    dy_i/dx_j = w_i * d(x_i / μ)/dx_j
    
    d(x_i / μ)/dx_j = (δ_ij * μ - x_i * dμ/dx_j) / μ^2
    
    dμ/dx_j = sign(x_j) / n
    
    So: d(x_i / μ)/dx_j = (δ_ij * μ - x_i * sign(x_j) / n) / μ^2
                        = δ_ij / μ - x_i * sign(x_j) / (n * μ^2)
    
    dL/dx_j = Σ_i (dL/dy_i * dy_i/dx_j)
            = Σ_i (dL/dy_i * w_i * (δ_ij / μ - x_i * sign(x_j) / (n * μ^2)))
            = dL/dy_j * w_j / μ - sign(x_j) / (n * μ^2) * Σ_i (dL/dy_i * w_i * x_i)
            = dL/dy_j * w_j / μ - sign(x_j) * C / (n * μ)
    
    where C = (1/μ) * Σ_i (dL/dy_i * w_i * x_i) = Σ_i (dL/dy_i * y_i)
    """
    row_idx = tl.program_id(0)
    
    grad_out_row = grad_output_ptr + row_idx * stride
    input_row = input_ptr + row_idx * stride
    grad_in_row = grad_input_ptr + row_idx * stride
    
    # Load precomputed mean
    mean_abs = tl.load(mean_ptr + row_idx).to(tl.float32)
    inv_mean = 1.0 / (mean_abs + eps)
    
    # First pass: compute C = Σ_i (dL/dy_i * y_i) = Σ_i (dL/dy_i * w_i * x_i / μ)
    C = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        grad_out = tl.load(grad_out_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(input_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        
        # y_i = x_i * inv_mean * w_i
        y = x * inv_mean * w
        C += tl.sum(tl.where(mask, grad_out * y, 0.0))
    
    # Second pass: compute gradients
    correction = C / n_cols
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        grad_out = tl.load(grad_out_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(input_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        
        sign_x = tl.where(x >= 0, 1.0, -1.0)
        
        # dL/dx_j = dL/dy_j * w_j / μ - sign(x_j) * C / (n * μ)
        grad_in = grad_out * w * inv_mean - sign_x * correction * inv_mean
        
        tl.store(grad_in_row + col_offsets, grad_in.to(grad_input_ptr.dtype.element_ty), mask=mask)


# =============================================================================
# AUTOGRAD FUNCTIONS
# =============================================================================

def _get_block_size(n: int) -> int:
    """Get optimal block size for dimension n."""
    return min(triton.next_power_of_2(n), 1024)


class RationalSoftmax(torch.autograd.Function):
    """Autograd function for rational softmax."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
        assert x.is_cuda, "Input must be on CUDA"
        
        original_shape = x.shape
        x_2d = x.contiguous().view(-1, x.shape[-1])
        n_rows, n_cols = x_2d.shape
        
        output = torch.empty_like(x_2d)
        
        BLOCK_SIZE = _get_block_size(n_cols)
        
        rational_softmax_fwd_kernel[(n_rows,)](
            x_2d, output,
            n_cols,
            x_2d.stride(0), output.stride(0),
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        output = output.view(original_shape)
        ctx.save_for_backward(x, output)
        ctx.eps = eps
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, output = ctx.saved_tensors
        eps = ctx.eps
        
        original_shape = x.shape
        x_2d = x.contiguous().view(-1, x.shape[-1])
        output_2d = output.contiguous().view(-1, output.shape[-1])
        grad_output_2d = grad_output.contiguous().view(-1, grad_output.shape[-1])
        n_rows, n_cols = x_2d.shape
        
        grad_input = torch.empty_like(x_2d)
        
        BLOCK_SIZE = _get_block_size(n_cols)
        
        rational_softmax_bwd_kernel[(n_rows,)](
            grad_output_2d, x_2d, output_2d, grad_input,
            n_cols,
            x_2d.stride(0),
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return grad_input.view(original_shape), None


class RationalSwiGLU(torch.autograd.Function):
    """Autograd function for rational SwiGLU."""
    
    @staticmethod
    def forward(ctx, gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        assert gate.is_cuda and value.is_cuda, "Inputs must be on CUDA"
        
        original_shape = gate.shape
        gate_2d = gate.contiguous().view(-1, gate.shape[-1])
        value_2d = value.contiguous().view(-1, value.shape[-1])
        n_rows, n_cols = gate_2d.shape
        
        output = torch.empty_like(gate_2d)
        
        BLOCK_SIZE = _get_block_size(n_cols)
        
        swiglu_fwd_kernel[(n_rows,)](
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
        
        original_shape = gate.shape
        gate_2d = gate.contiguous().view(-1, gate.shape[-1])
        value_2d = value.contiguous().view(-1, value.shape[-1])
        grad_output_2d = grad_output.contiguous().view(-1, grad_output.shape[-1])
        n_rows, n_cols = gate_2d.shape
        
        grad_gate = torch.empty_like(gate_2d)
        grad_value = torch.empty_like(value_2d)
        
        BLOCK_SIZE = _get_block_size(n_cols)
        
        swiglu_bwd_kernel[(n_rows,)](
            grad_output_2d, gate_2d, value_2d, grad_gate, grad_value,
            n_cols,
            gate_2d.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return grad_gate.view(original_shape), grad_value.view(original_shape)


class MeanErrorNorm(torch.autograd.Function):
    """Autograd function for mean-error normalization."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
        assert x.is_cuda, "Input must be on CUDA"
        
        original_shape = x.shape
        x_2d = x.contiguous().view(-1, x.shape[-1])
        n_rows, n_cols = x_2d.shape
        
        output = torch.empty_like(x_2d)
        mean = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        
        BLOCK_SIZE = _get_block_size(n_cols)
        
        mean_error_norm_fwd_kernel[(n_rows,)](
            x_2d, weight, output, mean,
            n_cols,
            x_2d.stride(0), output.stride(0),
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        ctx.save_for_backward(x, weight, mean)
        ctx.eps = eps
        
        return output.view(original_shape)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, mean = ctx.saved_tensors
        eps = ctx.eps
        
        original_shape = x.shape
        x_2d = x.contiguous().view(-1, x.shape[-1])
        grad_output_2d = grad_output.contiguous().view(-1, grad_output.shape[-1])
        n_rows, n_cols = x_2d.shape
        
        grad_input = torch.empty_like(x_2d)
        
        BLOCK_SIZE = _get_block_size(n_cols)
        
        mean_error_norm_bwd_kernel[(n_rows,)](
            grad_output_2d, x_2d, weight, mean, grad_input,
            n_cols,
            x_2d.stride(0),
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Compute grad_weight: sum over batch dimension
        # y = x * inv_mean * w => dy/dw = x * inv_mean
        inv_mean = 1.0 / (mean.unsqueeze(1) + eps)
        y = x_2d * inv_mean  # normalized x
        grad_weight = (grad_output_2d * y).sum(dim=0)
        
        return grad_input.view(original_shape), grad_weight, None


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
        sigma = 0.5 * (x / (abs_x + 1.0) + 1.0)
        sigma4 = sigma.pow(4)
        return sigma4 / (sigma4.sum(dim=-1, keepdim=True) + eps)


def rational_swiglu(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply rational SwiGLU activation."""
    if gate.is_cuda:
        return RationalSwiGLU.apply(gate, value)
    else:
        # CPU fallback
        abs_gate = gate.abs()
        sigma = 0.5 * (gate / (abs_gate + 1.0) + 1.0)
        return gate * sigma * value


def mean_error_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Apply mean-error normalization."""
    if x.is_cuda:
        return MeanErrorNorm.apply(x, weight, eps)
    else:
        # CPU fallback
        mean_abs = x.abs().mean(dim=-1, keepdim=True)
        return (x / (mean_abs + eps)) * weight
