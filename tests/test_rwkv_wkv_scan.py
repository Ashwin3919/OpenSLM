"""Tests: parallel WKV scan is numerically equivalent to sequential scan.

The parallel scan (_wkv_parallel) replaces the sequential Python loop
(_wkv_forward) by solving the linear recurrence

    state[t] = alpha * state[t-1] + b[t-1]   (alpha = exp_w, constant)

in closed form as an exponentially-decayed cumulative sum.  These tests
verify correctness (forward and gradient), cover edge cases, and confirm
the full RWKV model still trains with the new scan.
"""

import pytest
import torch

from src.models.rwkv.model import _wkv_forward, _wkv_parallel, RWKV_TimeMix
from src.models.rwkv.config import RWKVConfig


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_inputs(B: int, T: int, H: int, D: int, seed: int = 0):
    """Reproducible WKV kernel inputs on CPU."""
    torch.manual_seed(seed)
    k = torch.randn(B, T, H, D) * 0.1
    v = torch.randn(B, T, H, D) * 0.1
    # Typical model init: time_decay ≈ randn-5, so alpha ≈ exp(-exp(-5)) ≈ 0.993
    time_decay = torch.randn(H) - 5.0
    time_first = torch.randn(H)
    exp_w = torch.exp(-torch.exp(time_decay)).view(1, H, 1)
    exp_u = torch.exp(time_first).view(1, H, 1)
    return k, v, exp_w, exp_u


# ─── forward equivalence ──────────────────────────────────────────────────────

@pytest.mark.parametrize("B,T,H,D", [
    (1, 1, 2, 4),     # T=1 edge case: no past context, state is always zero
    (1, 2, 2, 4),     # T=2 edge case: one past token
    (2, 4, 2, 4),     # small, easy to trace by hand
    (2, 8, 4, 8),     # medium
    (4, 32, 4, 16),   # larger sequence
    (4, 128, 8, 64),  # full block_size used in default configs
])
def test_parallel_matches_sequential(B, T, H, D):
    """_wkv_parallel output is numerically equal to _wkv_forward (sequential)."""
    k, v, exp_w, exp_u = _make_inputs(B, T, H, D)

    out_seq = _wkv_forward(k, v, exp_w, exp_u)
    out_par = _wkv_parallel(k, v, exp_w, exp_u)

    assert out_par.shape == out_seq.shape, (
        f"Shape mismatch: parallel {out_par.shape} vs sequential {out_seq.shape}"
    )
    torch.testing.assert_close(
        out_par, out_seq, atol=1e-5, rtol=1e-4,
        msg=f"Mismatch for B={B} T={T} H={H} D={D}",
    )


def test_t1_state_is_zero():
    """At T=1 there is no past, so state_a and state_b must be zero.

    Both scans should therefore produce identical output equal to
    (exp_u * exp_k[0] * v[0]) / (exp_u * exp_k[0]).sum().
    """
    k, v, exp_w, exp_u = _make_inputs(B=3, T=1, H=4, D=8)
    out_seq = _wkv_forward(k, v, exp_w, exp_u)
    out_par = _wkv_parallel(k, v, exp_w, exp_u)
    torch.testing.assert_close(out_par, out_seq, atol=1e-6, rtol=1e-5)


def test_output_shape_unchanged():
    """Output shape must equal input (B, T, H, D)."""
    B, T, H, D = 3, 16, 4, 16
    k, v, exp_w, exp_u = _make_inputs(B, T, H, D)
    out = _wkv_parallel(k, v, exp_w, exp_u)
    assert out.shape == (B, T, H, D)


def test_output_is_finite():
    """No NaN or Inf in the parallel scan output."""
    k, v, exp_w, exp_u = _make_inputs(B=4, T=128, H=8, D=64)
    out = _wkv_parallel(k, v, exp_w, exp_u)
    assert torch.isfinite(out).all(), "Parallel scan produced non-finite values"


# ─── strong decay / strong bonus edge cases ───────────────────────────────────

def test_strong_decay():
    """Strong but representable decay: alpha ≈ 0.066 (time_decay=1.0).

    time_decay=1.0 → alpha = exp(-exp(1.0)) ≈ 0.066.  alpha^16 ≈ 1.3e-19
    which is well within float32 range.  Matches the sequential scan exactly.

    Note: pathologically large time_decay (>≈4.5 for T=16) causes float32
    underflow and is not representable. That regime never occurs in practice
    because time_decay is initialised as randn-5 (α ≈ 0.993).
    """
    B, T, H, D = 2, 16, 2, 8
    torch.manual_seed(7)
    k = torch.randn(B, T, H, D) * 0.1
    v = torch.randn(B, T, H, D) * 0.1
    # time_decay=1.0 → alpha ≈ 0.066 — strong decay, float32-safe for T=16
    exp_w = torch.exp(-torch.exp(torch.tensor([1.0, 1.0]))).view(1, H, 1)
    exp_u = torch.exp(torch.randn(H)).view(1, H, 1)

    out_seq = _wkv_forward(k, v, exp_w, exp_u)
    out_par = _wkv_parallel(k, v, exp_w, exp_u)
    torch.testing.assert_close(out_par, out_seq, atol=1e-5, rtol=1e-4)


def test_weak_decay():
    """Very weak decay (alpha ≈ 1): all past tokens contribute equally."""
    B, T, H, D = 2, 16, 2, 8
    torch.manual_seed(9)
    k = torch.randn(B, T, H, D) * 0.1
    v = torch.randn(B, T, H, D) * 0.1
    # small time_decay → exp(-exp(small)) ≈ exp(-tiny) ≈ 1 (weak decay)
    exp_w = torch.exp(-torch.exp(torch.tensor([-8.0, -8.0]))).view(1, H, 1)
    exp_u = torch.exp(torch.randn(H)).view(1, H, 1)

    out_seq = _wkv_forward(k, v, exp_w, exp_u)
    out_par = _wkv_parallel(k, v, exp_w, exp_u)
    torch.testing.assert_close(out_par, out_seq, atol=1e-5, rtol=1e-4)


# ─── gradient tests ───────────────────────────────────────────────────────────

def test_parallel_gradients_flow():
    """Backward pass through _wkv_parallel must produce valid gradients for k and v."""
    k, v, exp_w, exp_u = _make_inputs(B=2, T=8, H=4, D=8)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)

    out = _wkv_parallel(k, v, exp_w, exp_u)
    out.sum().backward()

    assert k.grad is not None, "k.grad is None after backward"
    assert v.grad is not None, "v.grad is None after backward"
    assert torch.isfinite(k.grad).all(), "k.grad contains non-finite values"
    assert torch.isfinite(v.grad).all(), "v.grad contains non-finite values"


def test_parallel_gradients_match_sequential():
    """Gradients of k and v must be numerically equal for both scan implementations."""
    k, v, exp_w, exp_u = _make_inputs(B=2, T=8, H=4, D=8)

    # Sequential path
    k1 = k.clone().detach().requires_grad_(True)
    v1 = v.clone().detach().requires_grad_(True)
    _wkv_forward(k1, v1, exp_w, exp_u).sum().backward()

    # Parallel path
    k2 = k.clone().detach().requires_grad_(True)
    v2 = v.clone().detach().requires_grad_(True)
    _wkv_parallel(k2, v2, exp_w, exp_u).sum().backward()

    torch.testing.assert_close(k2.grad, k1.grad, atol=1e-4, rtol=1e-3,
                                msg="k gradients differ between scan paths")
    torch.testing.assert_close(v2.grad, v1.grad, atol=1e-4, rtol=1e-3,
                                msg="v gradients differ between scan paths")


# ─── TimeMix integration ──────────────────────────────────────────────────────

@pytest.fixture
def tiny_rwkv_cfg():
    return RWKVConfig(
        vocab_size=100, block_size=16, n_layer=2, n_embd=32, n_head=4,
        ffn_mult=2, dropout=0.0,
    )


def test_time_mix_fast_matches_sequential(tiny_rwkv_cfg):
    """RWKV_TimeMix._wkv_fast and _wkv_sequential produce identical outputs."""
    torch.manual_seed(0)
    tm = RWKV_TimeMix(tiny_rwkv_cfg)
    tm.eval()

    B, T, C = 2, tiny_rwkv_cfg.block_size, tiny_rwkv_cfg.n_embd
    k = torch.randn(B, T, C) * 0.1
    v = torch.randn(B, T, C) * 0.1

    with torch.no_grad():
        out_fast = tm._wkv_fast(k.clone(), v.clone(), B, T, C)
        out_seq  = tm._wkv_sequential(k.clone(), v.clone(), B, T, C)

    torch.testing.assert_close(out_fast, out_seq, atol=1e-5, rtol=1e-4)


# ─── full-model smoke tests ───────────────────────────────────────────────────

def test_full_model_forward_finite(tiny_rwkv_cfg):
    """Full RWKV model forward pass (with parallel scan) produces finite outputs."""
    from src.models.rwkv.model import RWKVSLM
    torch.manual_seed(1)
    model = RWKVSLM(tiny_rwkv_cfg)
    model.eval()

    B, T = 2, tiny_rwkv_cfg.block_size
    idx = torch.randint(0, tiny_rwkv_cfg.vocab_size, (B, T))
    targets = torch.randint(0, tiny_rwkv_cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits, loss = model(idx, targets)

    assert logits.shape == (B, T, tiny_rwkv_cfg.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"


def test_full_model_forward_matches_sequential(tiny_rwkv_cfg):
    """Full model outputs are identical whether _wkv_fast or _wkv_sequential is used.

    We patch forward() on each TimeMix block to call _wkv_sequential, then
    compare logits with the default (parallel) forward.
    """
    import types
    from src.models.rwkv.model import RWKVSLM

    torch.manual_seed(2)
    model = RWKVSLM(tiny_rwkv_cfg)
    model.eval()

    B, T = 2, tiny_rwkv_cfg.block_size
    idx = torch.randint(0, tiny_rwkv_cfg.vocab_size, (B, T))

    # Default (parallel) forward
    with torch.no_grad():
        logits_par, _ = model(idx)

    # Patch every TimeMix to use sequential scan instead
    def _seq_forward(self, x):
        Bx, Tx, Cx = x.shape
        x_prev = torch.nn.functional.pad(x, (0, 0, 1, -1))
        k = self.key(x * self.mix_k + x_prev * (1.0 - self.mix_k))
        v = self.value(x * self.mix_v + x_prev * (1.0 - self.mix_v))
        r = self.receptance(x * self.mix_r + x_prev * (1.0 - self.mix_r))
        r = torch.sigmoid(r)
        wkv = self._wkv_sequential(k, v, Bx, Tx, Cx)
        return self.output(r * wkv)

    for block in model.blocks:
        block.time_mix.forward = types.MethodType(_seq_forward, block.time_mix)

    with torch.no_grad():
        logits_seq, _ = model(idx)

    torch.testing.assert_close(
        logits_par, logits_seq, atol=1e-5, rtol=1e-4,
        msg="Parallel and sequential scans produce different model outputs",
    )
