# Algebraic Transformer Triton Kernels

High-performance Triton kernels for the Algebraic Transformer architecture, featuring optimized forward and backward passes with purely rational (algebraic) operations.

## Architecture

```
kernels/
├── __init__.py          # Package exports
├── fused_ops.py         # Core fused operations (softmax, SwiGLU, norm)
├── forward.py           # Attention forward pass kernels
├── backward.py          # Attention backward pass kernels
├── algebraic_model.py   # Complete model implementation
└── tests.py             # Comprehensive tests and benchmarks
```

## Key Features

### Purely Algebraic Operations
No transcendental functions (exp, log, sqrt) - only rational operations:

1. **Rational Softmax**: `p_i = σ(x_i)^4 / Σ σ(x_j)^4`
   - Where `σ(x) = 0.5 * (x / (|x| + 1) + 1)`

2. **Rational SwiGLU**: `gate * σ(gate) * value`

3. **Mean-Error Normalization**: `x / mean(|x|) * γ`

### Memory-Efficient Attention
- Flash-attention style tiling for O(1) memory in sequence dimension
- Custom backward pass with recomputation (no stored attention matrices)
- Gradient checkpointing support

### Optimized Kernels
- Autotuned Triton kernels for different GPU architectures
- Fused operations to minimize memory bandwidth
- Both forward and backward passes fully implemented in Triton

## Usage

### Basic Training

```python
from kernels.algebraic_model import AlgebraicTransformerLM

# Create model
model = AlgebraicTransformerLM(
    vocab_size=32000,
    d_model=768,
    n_head=12,
    n_layers=12,
).cuda()

# Forward + backward
input_ids = torch.randint(0, 32000, (batch_size, seq_len)).cuda()
labels = torch.randint(0, 32000, (batch_size, seq_len)).cuda()

logits, loss = model(input_ids, labels=labels)
loss.backward()
```

### Using Individual Kernels

```python
from kernels import rational_attention, rational_softmax, rational_swiglu

# Attention with custom backward pass
output = rational_attention(q, k, v, alibi_slopes, scale, causal=True)

# Individual operations
probs = rational_softmax(scores)
hidden = rational_swiglu(gate, value)
```

### Pre-configured Model Sizes

```python
from kernels.algebraic_model import (
    create_small_model,   # ~125M params
    create_medium_model,  # ~350M params
    create_large_model,   # ~760M params
    create_xl_model,      # ~1.3B params
)

model = create_medium_model(vocab_size=50000).cuda()
```

## Kernel Details

### fused_ops.py

Core Triton kernels for individual operations:

| Kernel | Description | Backward Support |
|--------|-------------|------------------|
| `rational_softmax_fwd_kernel` | σ(x)^4 normalized softmax | ✓ |
| `rational_softmax_bwd_kernel` | Gradient computation | ✓ |
| `swiglu_fwd_kernel` | gate * σ(gate) * value | ✓ |
| `swiglu_bwd_kernel` | Gradients for gate and value | ✓ |
| `mean_error_norm_fwd_kernel` | x / mean(\|x\|) * γ | ✓ |
| `mean_error_norm_bwd_kernel` | Gradients for input and weight | ✓ |

### forward.py

Attention forward pass kernels:

| Kernel | Description | Use Case |
|--------|-------------|----------|
| `rational_attention_fwd_kernel` | Tiled attention with ALiBi | Long sequences (>1024) |
| `rational_attention_fwd_simple_kernel` | Non-tiled attention | Short sequences (≤1024) |
| `fused_qkv_proj_kernel` | Fused QKV projection | All sequences |

### backward.py

Attention backward pass kernels:

| Kernel | Description |
|--------|-------------|
| `rational_attention_bwd_dq_kernel` | Compute gradients w.r.t. Q |
| `rational_attention_bwd_dkv_kernel` | Compute gradients w.r.t. K, V |

The backward kernels **recompute** attention weights rather than storing them, following the Flash Attention approach for memory efficiency.

## Performance

Expected speedups over PyTorch reference implementation:

| Operation | Speedup | Memory Reduction |
|-----------|---------|------------------|
| Rational Softmax | 2-4x | 20-40% |
| Full Attention | 1.5-3x | 40-70% |
| Complete Model | 1.5-2.5x | 30-50% |

*Actual performance depends on GPU architecture, sequence length, and batch size.*

## Running Tests

```bash
python -m kernels.tests
```

This will run:
1. **Correctness tests**: Compare Triton outputs with PyTorch reference
2. **Gradient tests**: Verify gradients using `torch.autograd.gradcheck`
3. **Performance benchmarks**: Compare speed vs reference
4. **Memory tests**: Compare memory usage

## Requirements

- PyTorch >= 2.0
- Triton >= 2.0
- CUDA-capable GPU (compute capability 7.0+)

## Differences from Standard Transformers

| Component | Standard Transformer | Algebraic Transformer |
|-----------|---------------------|----------------------|
| Softmax | exp(x) / Σexp(x) | σ(x)^4 / Σσ(x)^4 |
| Activation | SiLU (x * sigmoid(x)) | x * σ(x) (rational) |
| Normalization | RMSNorm (x / √mean(x²)) | x / mean(\|x\|) |
| Position Encoding | RoPE / Learned | ALiBi |

## Gradient Flow

The rational softmax has different gradient properties than standard softmax:

- **Standard softmax**: `∂softmax/∂x = softmax * (I - softmax^T)`
- **Rational softmax**: `∂p/∂x = p * 4σ'(x)/σ(x) - p * Σ(p * 4σ'(x)/σ(x))`

The backward kernels handle these gradient computations efficiently through recomputation.

## Citation

If you use these kernels in your research, please cite:

```bibtex
@software{algebraic_transformer_kernels,
  title = {Algebraic Transformer Triton Kernels},
  year = {2024},
  description = {High-performance Triton kernels for algebraic transformer architecture}
}
```
