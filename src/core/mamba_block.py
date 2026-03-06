"""Mamba Selective State Space Block (pure PyTorch implementation).

Shared by ``src.models.mamba`` and ``src.models.jamba`` to avoid a
cross-plugin dependency.  This module contains only the Mamba block; the
surrounding layer structure (norms, residuals, embeddings) is owned by each
model plugin.

The key innovation of Mamba vs. classic SSMs is that the transition matrices
Δ, B, C are *input-dependent* (selected per token), making the scan selective.

Two scan paths are available:
- **Fast path** (auto-enabled): uses ``selective_scan_fn`` from the
  ``mamba-ssm`` package — a single fused CUDA parallel associative scan.
  Install with ``pip install mamba-ssm``.  Expected: ~15–20 it/s on A100.
- **Fallback path**: pure-PyTorch sequential loop — correct but slow
  (~1.3–1.6 it/s on A100 due to 12 layers × 128 sequential kernel launches).

Reference: Gu & Dao, 2023 — "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_fn
    _MAMBA_SSM_AVAILABLE = True
except ImportError:
    _MAMBA_SSM_AVAILABLE = False


class MambaBlock(nn.Module):
    """One Mamba SSM block (without surrounding norm / residual).

    Architecture::

        x_branch, z = split( Linear(d_model → 2*d_inner)(x) )
        x_branch → Conv1d → SiLU → SSM(x_branch)    # selective path
        z        → SiLU                               # gate path
        output   = Linear(d_inner → d_model)( SSM(x) * gate(z) )

    Args:
        d_model: Input / output hidden dimension.
        d_state: Dimension of the SSM latent state (H in the paper).
        d_conv: Kernel size of the depthwise convolution.
        expand: Expansion factor for the inner dimension
            (``d_inner = d_model * expand``).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection: d_model → 2 * d_inner  (x + z branches)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise 1-D convolution over the x branch
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # Projections for input-dependent SSM parameters (B, C, Δ)
        # Output: [B_param | C_param | log_dt]  — sizes d_state, d_state, 1
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Δ broadcast projection: scalar Δ per position → d_inner values
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Learnable SSM matrix A (log-parameterised for stability)
        # Shape: (d_inner, d_state)
        A = (
            torch.arange(1, d_state + 1, dtype=torch.float)
            .unsqueeze(0)
            .expand(self.d_inner, -1)
        )
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection scalar (D in the Mamba paper)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection: d_inner → d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ------------------------------------------------------------------
    # SSM helpers
    # ------------------------------------------------------------------

    def _ssm_sequential(self, x: torch.Tensor) -> torch.Tensor:
        """Pure-PyTorch sequential scan — fallback when mamba-ssm is not installed.

        Args:
            x: Convolved + activated inner tensor ``(B, T, d_inner)``.

        Returns:
            SSM output ``(B, T, d_inner)``.
        """
        B_batch, T, d_inner = x.shape
        A = -torch.exp(self.A_log)  # (d_inner, d_state) — negative for stability

        # Input-dependent parameters
        ssm_params = self.x_proj(x)                           # (B, T, 2*d_state + 1)
        B_param = ssm_params[:, :, : self.d_state]            # (B, T, d_state)
        C_param = ssm_params[:, :, self.d_state : 2 * self.d_state]
        dt_raw = ssm_params[:, :, -1:]                        # (B, T, 1)
        dt = F.softplus(self.dt_proj(dt_raw))                 # (B, T, d_inner)

        # Sequential scan: h_t = A_bar * h_{t-1} + B_bar * x_t
        h = torch.zeros(B_batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dt_t = dt[:, t]                                   # (B, d_inner)
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A)         # (B, d_inner, d_state)
            B_bar = dt_t.unsqueeze(-1) * B_param[:, t].unsqueeze(1)  # (B, d_inner, d_state)
            h = A_bar * h + B_bar * x[:, t].unsqueeze(-1)
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(-1)   # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                            # (B, T, d_inner)
        return y + self.D * x                                 # skip connection

    def _ssm_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """Fast SSM via mamba-ssm CUDA parallel associative scan.

        Replaces the sequential Python loop with a single fused CUDA kernel.
        Numerically equivalent output; ~10–20× faster on A100.

        Args:
            x: Convolved + activated inner tensor ``(B, T, d_inner)``.

        Returns:
            SSM output ``(B, T, d_inner)`` — includes D*x skip connection.
        """
        A = -torch.exp(self.A_log)                                    # (d_inner, d_state)
        ssm_params = self.x_proj(x)                                   # (B, T, 2*d_state+1)
        B_p = ssm_params[:, :, :self.d_state].transpose(1, 2).contiguous()          # (B, d_state, T)
        C_p = ssm_params[:, :, self.d_state:2*self.d_state].transpose(1, 2).contiguous()  # (B, d_state, T)
        dt = F.softplus(self.dt_proj(ssm_params[:, :, -1:]))          # (B, T, d_inner)
        dt = dt.transpose(1, 2).contiguous()                          # (B, d_inner, T)
        u = x.transpose(1, 2).contiguous()                            # (B, d_inner, T)
        # selective_scan_fn returns y + D*u (skip already included)
        y = _selective_scan_fn(u, dt, A, B_p, C_p, self.D)           # (B, d_inner, T)
        return y.transpose(1, 2)                                      # (B, T, d_inner)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch to CUDA fast path or pure-PyTorch fallback."""
        if _MAMBA_SSM_AVAILABLE:
            return self._ssm_cuda(x)
        return self._ssm_sequential(x)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one Mamba block.

        Args:
            x: Input tensor ``(B, T, d_model)``.

        Returns:
            Output tensor ``(B, T, d_model)``.
        """
        B, T, _ = x.shape

        xz = self.in_proj(x)                                  # (B, T, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)                    # each (B, T, d_inner)

        # Depthwise conv over the sequence dimension
        x_branch = x_branch.transpose(1, 2)                   # (B, d_inner, T)
        x_branch = self.conv1d(x_branch)[:, :, :T]           # trim causal padding
        x_branch = x_branch.transpose(1, 2)                   # (B, T, d_inner)
        x_branch = F.silu(x_branch)

        # Selective SSM
        y = self._ssm(x_branch)

        # Gated output
        output = y * F.silu(z)
        return self.out_proj(output)
