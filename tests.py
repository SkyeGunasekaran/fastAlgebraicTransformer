"""
Tests for Algebraic Transformer Kernels

Tests numerical correctness and gradient accuracy.
"""

import torch
import torch.nn.functional as F
import math
import sys

# Add parent directory for imports
sys.path.insert(0, '/home/claude')


# =============================================================================
# REFERENCE IMPLEMENTATIONS (from original.py)
# =============================================================================

def ref_rational_sigmoid(x):
    """Reference rational sigmoid."""
    return 0.5 * (x / (x.abs() + 1.0) + 1.0)


def ref_rational_softmax(x, eps=1e-6):
    """Reference rational softmax."""
    sigma = ref_rational_sigmoid(x)
    sigma4 = sigma.pow(4)
    return sigma4 / (sigma4.sum(dim=-1, keepdim=True) + eps)


def ref_rational_swiglu(gate, value):
    """Reference rational SwiGLU."""
    sigma = ref_rational_sigmoid(gate)
    return gate * sigma * value


def ref_mean_error_norm(x, weight, eps=1e-6):
    """Reference mean-error normalization."""
    mean_abs = x.abs().mean(dim=-1, keepdim=True)
    return (x / (mean_abs + eps)) * weight


def ref_rational_attention(q, k, v, alibi_slopes, scale, eps=1e-6):
    """Reference rational attention with ALiBi."""
    B, H, T, D = q.shape
    
    # Compute scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # ALiBi bias
    pos = torch.arange(T, device=q.device)
    rel_pos = pos[:, None] - pos[None, :]
    alibi = -rel_pos.abs().float() * alibi_slopes.view(1, H, 1, 1)
    scores = scores + alibi
    
    # Causal mask
    causal = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(causal[None, None, :, :], -1e4)
    
    # Rational softmax
    probs = ref_rational_softmax(scores, eps)
    
    # Output
    return torch.matmul(probs, v)


# =============================================================================
# CORRECTNESS TESTS
# =============================================================================

def test_rational_softmax():
    """Test rational softmax forward and backward."""
    from kernels.fused_ops import rational_softmax, RationalSoftmax
    
    print("Testing rational softmax...")
    
    for shape in [(2, 8, 64, 64), (4, 12, 128, 128), (2, 16, 256, 256)]:
        x = torch.randn(*shape, device='cuda', dtype=torch.float32, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        
        # Forward
        out = rational_softmax(x)
        ref = ref_rational_softmax(x_ref)
        
        fwd_diff = (out - ref).abs().max().item()
        sum_err = (out.sum(dim=-1) - 1.0).abs().max().item()
        
        # Backward
        grad = torch.randn_like(out)
        out.backward(grad)
        ref.backward(grad)
        
        bwd_diff = (x.grad - x_ref.grad).abs().max().item()
        
        fwd_ok = fwd_diff < 1e-5
        sum_ok = sum_err < 1e-5
        bwd_ok = bwd_diff < 1e-3
        
        status = "✓" if (fwd_ok and sum_ok and bwd_ok) else "✗"
        print(f"  {status} {shape}: fwd={fwd_diff:.2e}, sum_err={sum_err:.2e}, bwd={bwd_diff:.2e}")
    
    print()


def test_rational_swiglu():
    """Test rational SwiGLU forward and backward."""
    from kernels.fused_ops import rational_swiglu
    
    print("Testing rational SwiGLU...")
    
    for shape in [(2, 512, 768), (4, 1024, 1024), (8, 2048, 2048)]:
        gate = torch.randn(*shape, device='cuda', dtype=torch.float32, requires_grad=True)
        value = torch.randn(*shape, device='cuda', dtype=torch.float32, requires_grad=True)
        gate_ref = gate.detach().clone().requires_grad_(True)
        value_ref = value.detach().clone().requires_grad_(True)
        
        # Forward
        out = rational_swiglu(gate, value)
        ref = ref_rational_swiglu(gate_ref, value_ref)
        
        fwd_diff = (out - ref).abs().max().item()
        
        # Backward
        grad = torch.randn_like(out)
        out.backward(grad)
        ref.backward(grad)
        
        gate_grad_diff = (gate.grad - gate_ref.grad).abs().max().item()
        value_grad_diff = (value.grad - value_ref.grad).abs().max().item()
        
        fwd_ok = fwd_diff < 1e-5
        bwd_ok = max(gate_grad_diff, value_grad_diff) < 1e-3
        
        status = "✓" if (fwd_ok and bwd_ok) else "✗"
        print(f"  {status} {shape}: fwd={fwd_diff:.2e}, dgate={gate_grad_diff:.2e}, dval={value_grad_diff:.2e}")
    
    print()


def test_mean_error_norm():
    """Test mean-error normalization forward and backward."""
    from kernels.fused_ops import mean_error_norm
    
    print("Testing mean-error norm...")
    
    for shape in [(2, 512, 768), (4, 1024, 1024)]:
        x = torch.randn(*shape, device='cuda', dtype=torch.float32, requires_grad=True)
        weight = torch.randn(shape[-1], device='cuda', dtype=torch.float32, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        
        # Forward
        out = mean_error_norm(x, weight)
        ref = ref_mean_error_norm(x_ref, weight_ref)
        
        fwd_diff = (out - ref).abs().max().item()
        
        # Backward
        grad = torch.randn_like(out)
        out.backward(grad)
        ref.backward(grad)
        
        x_grad_diff = (x.grad - x_ref.grad).abs().max().item()
        w_grad_diff = (weight.grad - weight_ref.grad).abs().max().item()
        
        fwd_ok = fwd_diff < 1e-4
        bwd_ok = max(x_grad_diff, w_grad_diff) < 1e-3
        
        status = "✓" if (fwd_ok and bwd_ok) else "✗"
        print(f"  {status} {shape}: fwd={fwd_diff:.2e}, dx={x_grad_diff:.2e}, dw={w_grad_diff:.2e}")
    
    print()


def test_attention():
    """Test attention forward and backward."""
    from kernels.backward import rational_attention
    from kernels.forward import compute_alibi_slopes
    
    print("Testing rational attention...")
    
    for B, H, T, D in [(1, 4, 32, 32), (2, 8, 64, 64), (1, 4, 128, 32)]:
        q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
        
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        
        alibi_slopes = compute_alibi_slopes(H).cuda()
        scale = 1.0 / math.sqrt(D)
        
        # Forward
        out = rational_attention(q, k, v, alibi_slopes, scale)
        ref = ref_rational_attention(q_ref, k_ref, v_ref, alibi_slopes, scale)
        
        fwd_diff = (out - ref).abs().max().item()
        
        # Backward
        grad = torch.randn_like(out)
        out.backward(grad)
        ref.backward(grad)
        
        dq_diff = (q.grad - q_ref.grad).abs().max().item()
        dk_diff = (k.grad - k_ref.grad).abs().max().item()
        dv_diff = (v.grad - v_ref.grad).abs().max().item()
        
        fwd_ok = fwd_diff < 1e-3
        bwd_ok = max(dq_diff, dk_diff, dv_diff) < 1e-2
        
        status = "✓" if (fwd_ok and bwd_ok) else "✗"
        print(f"  {status} B={B},H={H},T={T},D={D}: fwd={fwd_diff:.2e}, dQ={dq_diff:.2e}, dK={dk_diff:.2e}, dV={dv_diff:.2e}")
    
    print()


def test_gradcheck():
    """Test gradients using torch.autograd.gradcheck."""
    from kernels.fused_ops import RationalSoftmax, RationalSwiGLU, MeanErrorNorm
    
    print("Testing with gradcheck (float64)...")
    
    # Softmax
    print("  Rational softmax...", end=" ")
    x = torch.randn(2, 4, 16, 16, device='cuda', dtype=torch.float64, requires_grad=True)
    try:
        passed = torch.autograd.gradcheck(
            lambda x: RationalSoftmax.apply(x, 1e-6),
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
        print("✓" if passed else "✗")
    except Exception as e:
        print(f"✗ ({e})")
    
    # SwiGLU
    print("  Rational SwiGLU...", end=" ")
    gate = torch.randn(2, 16, 32, device='cuda', dtype=torch.float64, requires_grad=True)
    value = torch.randn(2, 16, 32, device='cuda', dtype=torch.float64, requires_grad=True)
    try:
        passed = torch.autograd.gradcheck(
            RationalSwiGLU.apply,
            (gate, value),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
        print("✓" if passed else "✗")
    except Exception as e:
        print(f"✗ ({e})")
    
    # Mean-error norm
    print("  Mean-error norm...", end=" ")
    x = torch.randn(2, 16, 32, device='cuda', dtype=torch.float64, requires_grad=True)
    w = torch.randn(32, device='cuda', dtype=torch.float64, requires_grad=True)
    try:
        passed = torch.autograd.gradcheck(
            lambda x, w: MeanErrorNorm.apply(x, w, 1e-6),
            (x, w),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
        print("✓" if passed else "✗")
    except Exception as e:
        print(f"✗ ({e})")
    
    print()


def test_model():
    """Test full model forward and backward."""
    from kernels.algebraic_model import AlgebraicTransformerLM
    
    print("Testing full model...")
    
    model = AlgebraicTransformerLM(
        vocab_size=1000,
        d_model=256,
        n_head=4,
        n_layers=2,
        gradient_checkpointing=False,
    ).cuda()
    
    x = torch.randint(0, 1000, (2, 64), device='cuda')
    labels = torch.randint(0, 1000, (2, 64), device='cuda')
    
    # Forward
    logits, loss = model(x, labels=labels)
    print(f"  Forward: logits {logits.shape}, loss={loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    
    print(f"  Backward: grad_norm={grad_norm:.4f}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    print("=" * 70)
    print("ALGEBRAIC TRANSFORMER KERNEL TESTS")
    print("=" * 70)
    print()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    print("=" * 70)
    print("CORRECTNESS TESTS")
    print("=" * 70)
    print()
    
    test_rational_softmax()
    test_rational_swiglu()
    test_mean_error_norm()
    test_attention()
    
    print("=" * 70)
    print("GRADIENT CHECKS")
    print("=" * 70)
    print()
    
    test_gradcheck()
    
    print("=" * 70)
    print("MODEL TEST")
    print("=" * 70)
    print()
    
    test_model()
    
    print("=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
