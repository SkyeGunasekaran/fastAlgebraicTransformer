"""
Tests for Algebraic Transformer Kernels

Comprehensive tests for:
1. Numerical correctness (comparing Triton vs PyTorch reference)
2. Gradient correctness (using torch.autograd.gradcheck)
3. Performance benchmarks
"""

import torch
import torch.nn.functional as F
import time
from typing import Callable, Tuple
import math


def reference_rational_softmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference implementation of rational softmax in pure PyTorch."""
    abs_x = x.abs()
    s = x / (abs_x + 1.0)
    p_base = (s + 1.0) * 0.5
    p4 = p_base.pow(4)
    return p4 / (p4.sum(dim=-1, keepdim=True) + eps)


def reference_rational_swiglu(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Reference implementation of rational SwiGLU."""
    abs_gate = gate.abs()
    sigmoid_approx = 0.5 * (gate / (abs_gate + 1.0) + 1.0)
    return gate * sigmoid_approx * value


def reference_mean_error_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference implementation of mean-error normalization."""
    magnitude = x.abs().mean(dim=-1, keepdim=True)
    return (x / (magnitude + eps)) * weight


def reference_rational_attention(
    q: torch.Tensor,  # [B, H, T, D]
    k: torch.Tensor,
    v: torch.Tensor,
    alibi_slopes: torch.Tensor,  # [H]
    scale: float,
    causal: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference implementation of rational attention."""
    B, H, T, D = q.shape
    
    # Compute scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # ALiBi bias
    positions = torch.arange(T, device=q.device)
    relative_pos = positions[:, None] - positions[None, :]
    alibi_bias = -relative_pos.abs().float() * alibi_slopes.view(1, H, 1, 1)
    scores = scores + alibi_bias
    
    # Causal mask
    if causal:
        causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(causal_mask[None, None, :, :], -1e4)
    
    # Rational softmax
    probs = reference_rational_softmax(scores, eps)
    
    # Compute output
    return torch.matmul(probs, v)


# =============================================================================
# NUMERICAL CORRECTNESS TESTS
# =============================================================================

def test_rational_softmax_correctness():
    """Test that Triton rational softmax matches reference."""
    from kernels import rational_softmax
    
    print("Testing rational softmax correctness...")
    
    test_cases = [
        (2, 8, 64, 64),      # Small
        (4, 12, 256, 256),   # Medium
        (2, 16, 512, 1024),  # Large
    ]
    
    for B, H, T_q, T_k in test_cases:
        x = torch.randn(B, H, T_q, T_k, device='cuda', dtype=torch.float32)
        
        # Reference
        ref = reference_rational_softmax(x)
        
        # Triton
        out = rational_softmax(x)
        
        # Check
        max_diff = (ref - out).abs().max().item()
        mean_diff = (ref - out).abs().mean().item()
        
        # Verify sum to 1
        sum_check = out.sum(dim=-1)
        sum_error = (sum_check - 1.0).abs().max().item()
        
        status = "✓" if max_diff < 1e-4 and sum_error < 1e-5 else "✗"
        print(f"  {status} Shape {x.shape}: max_diff={max_diff:.2e}, sum_error={sum_error:.2e}")
    
    print()


def test_rational_swiglu_correctness():
    """Test that Triton SwiGLU matches reference."""
    from kernels import rational_swiglu
    
    print("Testing rational SwiGLU correctness...")
    
    test_cases = [
        (2, 512, 768),
        (4, 1024, 2048),
        (8, 2048, 4096),
    ]
    
    for B, T, D in test_cases:
        gate = torch.randn(B, T, D, device='cuda', dtype=torch.float32)
        value = torch.randn(B, T, D, device='cuda', dtype=torch.float32)
        
        ref = reference_rational_swiglu(gate, value)
        out = rational_swiglu(gate, value)
        
        max_diff = (ref - out).abs().max().item()
        status = "✓" if max_diff < 1e-5 else "✗"
        print(f"  {status} Shape ({B}, {T}, {D}): max_diff={max_diff:.2e}")
    
    print()


def test_mean_error_norm_correctness():
    """Test that Triton mean-error norm matches reference."""
    from kernels import mean_error_norm
    
    print("Testing mean-error norm correctness...")
    
    test_cases = [
        (2, 512, 768),
        (4, 1024, 1024),
        (8, 2048, 2048),
    ]
    
    for B, T, D in test_cases:
        x = torch.randn(B, T, D, device='cuda', dtype=torch.float32)
        weight = torch.randn(D, device='cuda', dtype=torch.float32)
        
        ref = reference_mean_error_norm(x, weight)
        out = mean_error_norm(x, weight)
        
        max_diff = (ref - out).abs().max().item()
        status = "✓" if max_diff < 1e-4 else "✗"
        print(f"  {status} Shape ({B}, {T}, {D}): max_diff={max_diff:.2e}")
    
    print()


def test_attention_correctness():
    """Test that Triton attention matches reference."""
    from kernels import rational_attention, compute_alibi_slopes
    
    print("Testing rational attention correctness...")
    
    test_cases = [
        (1, 4, 64, 32),    # Small
        (2, 8, 128, 64),   # Medium
        (2, 12, 256, 64),  # Larger
    ]
    
    for B, H, T, D in test_cases:
        q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
        alibi_slopes = compute_alibi_slopes(H).cuda()
        scale = 1.0 / math.sqrt(D)
        
        # Reference (with gradients)
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        ref_out = reference_rational_attention(q_ref, k_ref, v_ref, alibi_slopes, scale)
        
        # Triton
        triton_out = rational_attention(q, k, v, alibi_slopes, scale)
        
        # Forward check
        max_diff = (ref_out - triton_out).abs().max().item()
        status = "✓" if max_diff < 1e-3 else "✗"
        print(f"  {status} Shape ({B}, {H}, {T}, {D}): forward max_diff={max_diff:.2e}")
        
        # Backward check
        grad_out = torch.randn_like(triton_out)
        
        ref_out.backward(grad_out)
        triton_out.backward(grad_out)
        
        grad_q_diff = (q.grad - q_ref.grad).abs().max().item()
        grad_k_diff = (k.grad - k_ref.grad).abs().max().item()
        grad_v_diff = (v.grad - v_ref.grad).abs().max().item()
        
        grad_status = "✓" if max(grad_q_diff, grad_k_diff, grad_v_diff) < 1e-2 else "✗"
        print(f"    {grad_status} Gradients: dQ={grad_q_diff:.2e}, dK={grad_k_diff:.2e}, dV={grad_v_diff:.2e}")
    
    print()


# =============================================================================
# GRADIENT TESTS
# =============================================================================

def test_gradients():
    """Test gradient correctness using finite differences."""
    from kernels import RationalSoftmax, RationalSwiGLU, MeanErrorNorm
    
    print("Testing gradients with gradcheck...")
    
    # Rational softmax gradients
    print("  Rational softmax...")
    x = torch.randn(2, 4, 32, 32, device='cuda', dtype=torch.float64, requires_grad=True)
    
    def softmax_fn(x):
        return RationalSoftmax.apply(x, 1e-6)
    
    try:
        passed = torch.autograd.gradcheck(softmax_fn, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)
        print(f"    {'✓' if passed else '✗'} Gradcheck passed")
    except Exception as e:
        print(f"    ✗ Gradcheck failed: {e}")
    
    # SwiGLU gradients
    print("  Rational SwiGLU...")
    gate = torch.randn(2, 32, 64, device='cuda', dtype=torch.float64, requires_grad=True)
    value = torch.randn(2, 32, 64, device='cuda', dtype=torch.float64, requires_grad=True)
    
    def swiglu_fn(gate, value):
        return RationalSwiGLU.apply(gate, value)
    
    try:
        passed = torch.autograd.gradcheck(swiglu_fn, (gate, value), eps=1e-6, atol=1e-4, rtol=1e-3)
        print(f"    {'✓' if passed else '✗'} Gradcheck passed")
    except Exception as e:
        print(f"    ✗ Gradcheck failed: {e}")
    
    # Mean-error norm gradients
    print("  Mean-error norm...")
    x = torch.randn(2, 32, 64, device='cuda', dtype=torch.float64, requires_grad=True)
    weight = torch.randn(64, device='cuda', dtype=torch.float64, requires_grad=True)
    
    def norm_fn(x, weight):
        return MeanErrorNorm.apply(x, weight, 1e-6)
    
    try:
        passed = torch.autograd.gradcheck(norm_fn, (x, weight), eps=1e-6, atol=1e-4, rtol=1e-3)
        print(f"    {'✓' if passed else '✗'} Gradcheck passed")
    except Exception as e:
        print(f"    ✗ Gradcheck failed: {e}")
    
    print()


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = 10,
    iterations: int = 100,
) -> Tuple[float, float]:
    """Benchmark a function, returning mean and std time in ms."""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = fn(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    
    return mean, std


def benchmark_softmax():
    """Benchmark rational softmax vs reference."""
    from kernels import rational_softmax
    
    print("Benchmarking rational softmax...")
    print(f"{'Shape':<30} {'Reference':>15} {'Triton':>15} {'Speedup':>10}")
    print("-" * 75)
    
    test_cases = [
        (4, 12, 512, 512),
        (4, 12, 1024, 1024),
        (4, 12, 2048, 2048),
        (2, 16, 4096, 4096),
    ]
    
    for shape in test_cases:
        x = torch.randn(*shape, device='cuda', dtype=torch.float16)
        
        ref_time, ref_std = benchmark_fn(reference_rational_softmax, x.float())
        triton_time, triton_std = benchmark_fn(rational_softmax, x)
        
        speedup = ref_time / triton_time
        print(f"{str(shape):<30} {ref_time:>12.2f}ms {triton_time:>12.2f}ms {speedup:>9.2f}x")
    
    print()


def benchmark_attention():
    """Benchmark rational attention vs reference."""
    from kernels import rational_attention, compute_alibi_slopes
    
    print("Benchmarking rational attention...")
    print(f"{'Config':<25} {'Reference':>15} {'Triton':>15} {'Speedup':>10}")
    print("-" * 70)
    
    test_cases = [
        (2, 12, 512, 64),
        (2, 12, 1024, 64),
        (2, 12, 2048, 64),
        (4, 16, 1024, 64),
    ]
    
    for B, H, T, D in test_cases:
        q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16)
        alibi_slopes = compute_alibi_slopes(H).cuda()
        scale = 1.0 / math.sqrt(D)
        
        def ref_fn():
            return reference_rational_attention(q.float(), k.float(), v.float(), alibi_slopes, scale)
        
        def triton_fn():
            return rational_attention(q, k, v, alibi_slopes, scale)
        
        ref_time, _ = benchmark_fn(ref_fn)
        triton_time, _ = benchmark_fn(triton_fn)
        
        speedup = ref_time / triton_time
        config = f"B={B}, H={H}, T={T}, D={D}"
        print(f"{config:<25} {ref_time:>12.2f}ms {triton_time:>12.2f}ms {speedup:>9.2f}x")
    
    print()


def benchmark_full_model():
    """Benchmark full model forward + backward."""
    from kernels.algebraic_model import AlgebraicTransformerLM
    
    print("Benchmarking full model forward + backward...")
    print(f"{'Config':<35} {'Forward':>12} {'Backward':>12} {'Total':>12}")
    print("-" * 75)
    
    configs = [
        (768, 12, 6, 512),    # Small
        (768, 12, 12, 512),   # Medium
        (1024, 16, 12, 512),  # Large
    ]
    
    for d_model, n_head, n_layers, seq_len in configs:
        model = AlgebraicTransformerLM(
            vocab_size=32000,
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            gradient_checkpointing=True,
        ).cuda().half()
        
        x = torch.randint(0, 32000, (2, seq_len), device='cuda')
        labels = torch.randint(0, 32000, (2, seq_len), device='cuda')
        
        # Warmup
        for _ in range(3):
            logits, loss = model(x, labels=labels)
            loss.backward()
            model.zero_grad()
        
        torch.cuda.synchronize()
        
        # Forward
        times_fwd = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits, loss = model(x, labels=labels)
            torch.cuda.synchronize()
            times_fwd.append((time.perf_counter() - start) * 1000)
        
        # Backward
        times_bwd = []
        for _ in range(10):
            logits, loss = model(x, labels=labels)
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            times_bwd.append((time.perf_counter() - start) * 1000)
            model.zero_grad()
        
        fwd = sum(times_fwd) / len(times_fwd)
        bwd = sum(times_bwd) / len(times_bwd)
        
        config = f"d={d_model}, h={n_head}, L={n_layers}"
        print(f"{config:<35} {fwd:>10.2f}ms {bwd:>10.2f}ms {fwd+bwd:>10.2f}ms")
        
        del model
        torch.cuda.empty_cache()
    
    print()


# =============================================================================
# MEMORY USAGE TESTS
# =============================================================================

def test_memory_usage():
    """Compare memory usage between reference and Triton implementations."""
    from kernels import rational_attention, compute_alibi_slopes
    
    print("Testing memory usage...")
    print(f"{'Config':<30} {'Reference':>15} {'Triton':>15} {'Reduction':>12}")
    print("-" * 75)
    
    test_cases = [
        (2, 12, 1024, 64),
        (2, 12, 2048, 64),
        (2, 12, 4096, 64),
    ]
    
    for B, H, T, D in test_cases:
        # Reference memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16, requires_grad=True)
        alibi_slopes = compute_alibi_slopes(H).cuda()
        scale = 1.0 / math.sqrt(D)
        
        out = reference_rational_attention(q.float(), k.float(), v.float(), alibi_slopes, scale)
        out.sum().backward()
        
        torch.cuda.synchronize()
        ref_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        del q, k, v, out
        torch.cuda.empty_cache()
        
        # Triton memory
        torch.cuda.reset_peak_memory_stats()
        
        q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16, requires_grad=True)
        
        out = rational_attention(q, k, v, alibi_slopes, scale)
        out.sum().backward()
        
        torch.cuda.synchronize()
        triton_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        reduction = (1 - triton_mem / ref_mem) * 100
        config = f"B={B}, H={H}, T={T}, D={D}"
        print(f"{config:<30} {ref_mem:>12.1f}MB {triton_mem:>12.1f}MB {reduction:>10.1f}%")
        
        del q, k, v, out
        torch.cuda.empty_cache()
    
    print()


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("ALGEBRAIC TRANSFORMER KERNEL TESTS")
    print("=" * 80)
    print()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Tests require GPU.")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    # Run tests
    print("=" * 80)
    print("CORRECTNESS TESTS")
    print("=" * 80)
    print()
    
    test_rational_softmax_correctness()
    test_rational_swiglu_correctness()
    test_mean_error_norm_correctness()
    test_attention_correctness()
    
    print("=" * 80)
    print("GRADIENT TESTS")
    print("=" * 80)
    print()
    
    test_gradients()
    
    print("=" * 80)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 80)
    print()
    
    benchmark_softmax()
    benchmark_attention()
    benchmark_full_model()
    
    print("=" * 80)
    print("MEMORY USAGE")
    print("=" * 80)
    print()
    
    test_memory_usage()
    
    print("=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
