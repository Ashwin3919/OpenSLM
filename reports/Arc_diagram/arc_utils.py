"""
Shared drawing utilities for OpenSLM architecture diagrams.

Defines:
  COLORS    — consistent colour palette used across all 8 diagrams
  Diagram   — class that manages the vertical-flow layout and drawing
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "embed":  "#4A90D9",   # blue       — token / position embedding
    "norm":   "#E8A838",   # amber      — LayerNorm / RMSNorm
    "attn":   "#4EAD7B",   # teal       — causal self-attention
    "ssm":    "#5BA3C9",   # steel-blue — Mamba SSM scan
    "ffn":    "#C75F7A",   # rose       — MLP / FFN / SwiGLU
    "gate":   "#D4855A",   # orange     — gating / receptance (RWKV)
    "moe":    "#6ABBA8",   # seafoam    — MoE routing / shared expert
    "out":    "#8E6BB5",   # violet     — LM head / output
    "conv":   "#7B9E5A",   # olive      — depthwise convolution
    "wkv":    "#5B8AC9",   # periwinkle — WKV recurrence (RWKV)
    "ret":    "#4EAD7B",   # teal       — retention (same family as attn)
}

_BG_BLOCK  = "#F0F2F5"
_ED_BLOCK  = "#AABBCC"
_TEXT_FG   = "white"
_TEXT_BG   = "#1C1C1C"

FIG_W    = 5.4
BOX_W    = 3.4
BOX_H_LG = 0.72   # attention, SSM, FFN
BOX_H_MD = 0.55   # medium-height boxes
BOX_H_SM = 0.42   # LayerNorm / RMSNorm
_GAP     = 0.20   # gap between component boxes (unused — arrow absorbs it)
_ARR_GAP = 0.20   # vertical height of each arrow

_PLOTS = Path(__file__).parent / "plots"


# ---------------------------------------------------------------------------
# Diagram class
# ---------------------------------------------------------------------------
class Diagram:
    """
    Manages the y-coordinate pen and provides drawing helpers for
    a top-to-bottom vertical-flow architecture diagram.
    """

    def __init__(self, title: str, fig_h: float = 14.0):
        self._fig_h = fig_h
        self.fig, self.ax = plt.subplots(figsize=(FIG_W, fig_h))
        self.ax.set_xlim(0, FIG_W)
        self.ax.set_ylim(0, fig_h)
        self.ax.axis("off")
        self.fig.patch.set_facecolor("white")
        self.ax.set_facecolor("white")
        self.cx = FIG_W / 2
        self.y  = fig_h - 0.45      # pen starts near the top
        self._blk_top: float   = 0.0
        self._blk_label: str   = ""
        # title
        self.ax.text(
            self.cx, self.y, title,
            ha="center", va="top",
            fontsize=12.5, fontweight="bold", color=_TEXT_BG,
        )
        self.y -= 0.58

    # ── low-level ───────────────────────────────────────────────────────────

    def _rect(self, cx, cy, w, h, color, zorder=3):
        r = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.06",
            linewidth=0, edgecolor="none",
            facecolor=color, zorder=zorder,
        )
        self.ax.add_patch(r)

    def _arrow(self, side_label: Optional[str] = None):
        y0, y1 = self.y, self.y - _ARR_GAP
        self.ax.annotate(
            "", xy=(self.cx, y1 + 0.025), xytext=(self.cx, y0 - 0.025),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.3),
        )
        if side_label:
            self.ax.text(
                self.cx + BOX_W / 2 + 0.10, (y0 + y1) / 2, side_label,
                ha="left", va="center", fontsize=7, color="#777", style="italic",
            )
        self.y = y1

    # ── public API ──────────────────────────────────────────────────────────

    def node(self, text: str):
        """Plain rounded-rect node — used for Input and Output."""
        self.ax.text(
            self.cx, self.y, text,
            ha="center", va="center", fontsize=9, color="#444",
            bbox=dict(
                facecolor="#F8F8F8", edgecolor="#CCC",
                boxstyle="round,pad=0.3", linewidth=1.0,
            ),
        )
        self.y -= 0.40
        self._arrow()

    def box(
        self,
        label: str,
        color: str,
        sub: Optional[str] = None,
        h: float = BOX_H_LG,
        residual: bool = False,
        arrow: bool = True,
    ):
        """Solid coloured rounded-rect component box."""
        cy = self.y - h / 2
        self._rect(self.cx, cy, BOX_W, h, color)
        # main label (shift up slightly when there is a sub-label)
        ty = cy if sub is None else cy + h * 0.14
        self.ax.text(
            self.cx, ty, label,
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=_TEXT_FG, zorder=4,
        )
        if sub:
            self.ax.text(
                self.cx, cy - h * 0.22, sub,
                ha="center", va="center",
                fontsize=7.1, color=_TEXT_FG, alpha=0.88, zorder=4,
            )
        self.y = cy - h / 2
        if residual:
            self.ax.text(
                self.cx + BOX_W / 2 + 0.08, self.y + 0.04,
                "+  residual",
                ha="left", va="center", fontsize=7.2, color="#888",
            )
        if arrow:
            self._arrow()

    def begin_block(self, label: str = "× n_layer"):
        """Record the top edge of a repeating block."""
        self._blk_top   = self.y + 0.15
        self._blk_label = label

    def end_block(self):
        """Draw the dashed border around the repeating block."""
        m  = 0.18
        x0 = self.cx - BOX_W / 2 - m
        y0 = self.y - 0.10
        bw = BOX_W + 2 * m
        bh = self._blk_top - y0
        r = FancyBboxPatch(
            (x0, y0), bw, bh,
            boxstyle="round,pad=0.06",
            linewidth=1.5, edgecolor=_ED_BLOCK,
            facecolor=_BG_BLOCK, linestyle="--", zorder=1,
        )
        self.ax.add_patch(r)
        self.ax.text(
            x0 + bw - 0.08, y0 + bh / 2, self._blk_label,
            ha="right", va="center", fontsize=8.5,
            color="#667788", fontweight="bold",
            bbox=dict(
                facecolor="white", edgecolor=_ED_BLOCK,
                boxstyle="round,pad=0.2", linewidth=0.8,
            ),
            zorder=4,
        )
        self.y = y0
        self._arrow()

    def legend(self, items: list[tuple[str, str]]):
        """items: [(label, color), ...]  — placed at bottom-left of axes."""
        patches = [mpatches.Patch(color=c, label=l) for l, c in items]
        self.ax.legend(
            handles=patches,
            bbox_to_anchor=(0.01, 0.01), loc="lower left",
            bbox_transform=self.ax.transAxes,
            fontsize=7.5, framealpha=0.92, ncol=2,
            handlelength=1.2, handleheight=0.9, borderpad=0.6,
        )

    def param_note(self, text: str):
        """Small config summary — bottom-right corner of axes."""
        self.ax.text(
            0.99, 0.01, text,
            transform=self.ax.transAxes,
            ha="right", va="bottom", fontsize=7.0, color="#555",
            bbox=dict(
                facecolor="#F5F5F5", edgecolor="#DDD",
                boxstyle="round,pad=0.3", linewidth=0.8,
            ),
        )

    def save(self, filename: str):
        """Trim y-axis to drawn content, save PNG to plots/ folder."""
        self.ax.set_ylim(self.y - 1.9, self._fig_h)
        _PLOTS.mkdir(exist_ok=True)
        path = _PLOTS / filename
        self.fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(self.fig)
        print(f"  Saved → {path}")
