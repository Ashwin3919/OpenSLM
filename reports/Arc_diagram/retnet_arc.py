import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def draw_box(ax, x, y, width, height, text, facecolor='white', edgecolor='black',
             fontsize=12, fontweight='normal', ha='center', va='center'):
    rect = patches.Rectangle((x - width/2, y - height/2), width, height,
                              linewidth=1.2, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, fontweight=fontweight,
            family='serif', zorder=3)
    return rect


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2, shrinkA=0, shrinkB=0),
                zorder=1)


plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(9, 11), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(-2, 11)
ax.axis('off')

center_x = 5.0
box_width = 5.0

# Input
ax.text(center_x, 10.2, r"Input token IDs $(B, T)$",
        ha='center', va='center', fontsize=12, family='serif')
draw_arrow(ax, center_x, 9.9, center_x, 9.3)

# Embedding Block  (no wpe — decay mask encodes position)
embed_y, embed_h = 8.8, 1.0
draw_box(ax, center_x, embed_y, box_width, embed_h,
         "wte: token embedding    (no wpe — decay mask encodes position)\nDropout")
ax.text(center_x + box_width/2 + 0.3, embed_y, r"$(B, T, n_{embd})$",
        ha='left', va='center', fontsize=12, family='serif')
draw_arrow(ax, center_x, embed_y - embed_h/2, center_x, 7.5)

# RetNetBlock
tf_y, tf_h = 5.1, 4.0
draw_box(ax, center_x, tf_y, box_width + 0.4, tf_h,
         "RetNetBlock\n\n"
         "RMSNorm\n"
         "MultiScaleRetention  (exponential decay mask)\n"
         "GroupNorm on retention output\n"
         "RMSNorm\n"
         "SwiGLU FFN",
         facecolor='#f8f9fa', fontsize=11)
ax.text(center_x + box_width/2 + 0.5, tf_y + tf_h/2 - 0.2, r"$\times n_{layer}$",
        ha='left', va='center', fontsize=14, family='serif')
ax.text(center_x + box_width/2 + 0.5, tf_y - 0.3, "residual connection\naround each",
        ha='left', va='center', fontsize=11, family='serif', style='italic', color='#444444')
draw_arrow(ax, center_x, tf_y - tf_h/2, center_x, 2.8)

# Final RMSNorm
ln_y, ln_h = 2.4, 0.8
draw_box(ax, center_x, ln_y, box_width, ln_h, "RMSNorm (final)")
draw_arrow(ax, center_x, ln_y - ln_h/2, center_x, 1.3)

# LM Head
head_y, head_h = 0.9, 0.8
draw_box(ax, center_x, head_y, box_width + 1.0, head_h,
         r"lm\_head: Linear  $(n_{embd} \rightarrow vocab\_size)$",
         fontsize=11)
draw_arrow(ax, center_x, head_y - head_h/2, center_x, 0.0)

# Logits
logits_y, logits_h = -0.5, 1.0
draw_box(ax, center_x, logits_y, box_width + 1.0, logits_h,
         r"Logits $(B, T, vocab\_size)$  [training — parallel mode]" + "\n" +
         r"Logits $(B, 1, vocab\_size)$  [generation]",
         fontsize=11)

# Weight-tying dashed line
tie_x_start = center_x - (box_width + 1.0) / 2
tie_x_outer = 1.0
wte_y = embed_y + 0.15
ax.plot([tie_x_start, tie_x_outer, tie_x_outer, center_x - box_width / 2],
        [head_y, head_y, wte_y, wte_y],
        linestyle='--', color='#555555', linewidth=1.2, zorder=1)
ax.text(tie_x_outer - 0.2, (head_y + wte_y) / 2, "weight-tied",
        rotation=90, ha='center', va='center', fontsize=11, family='serif', color='#555555')

out_dir = Path(__file__).parent / "plots"
out_dir.mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(out_dir / "retnet_architecture.png", format='png', bbox_inches='tight', dpi=300)
plt.close()
print(f"  Saved → {out_dir / 'retnet_architecture.png'}")
