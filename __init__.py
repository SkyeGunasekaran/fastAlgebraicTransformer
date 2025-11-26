"""
Algebraic Transformer Triton Kernels

High-performance kernels for transformers using purely algebraic operations.
"""

from .fused_ops import (
    rational_softmax,
    rational_swiglu, 
    mean_error_norm,
    RationalSoftmax,
    RationalSwiGLU,
    MeanErrorNorm,
    EPS,
)

from .forward import (
    rational_attention_forward,
    compute_alibi_slopes,
)

from .backward import (
    rational_attention,
    rational_attention_backward,
    RationalAttentionFunction,
)

from .algebraic_model import (
    AlgebraicTransformerLM,
    RationalMR,
    RationalSwiGLU as RationalSwiGLUModule,
    AlgebraicAttention,
    AlgebraicBlock,
    create_small_model,
    create_medium_model,
    create_large_model,
)

__all__ = [
    # Fused ops
    'rational_softmax',
    'rational_swiglu',
    'mean_error_norm',
    'RationalSoftmax',
    'RationalSwiGLU',
    'MeanErrorNorm',
    'EPS',
    # Forward
    'rational_attention_forward',
    'compute_alibi_slopes',
    # Backward
    'rational_attention',
    'rational_attention_backward', 
    'RationalAttentionFunction',
    # Model
    'AlgebraicTransformerLM',
    'RationalMR',
    'AlgebraicAttention',
    'AlgebraicBlock',
    'create_small_model',
    'create_medium_model',
    'create_large_model',
]
