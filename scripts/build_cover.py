#!/usr/bin/env python3
"""
build_cover.py — Generate a professional cover image for BioRefusalAudit.
Output: demo/cover.png (2100x1400, dark theme, 150 dpi)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "demo" / "data_for_viz.json"
OUT  = ROOT / "demo" / "cover.png"

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#060a12"
PANEL  = "#0b1020"
BORDER = "#1a2840"
CYAN   = "#00d4ff"
AMBER  = "#ffb300"
RED    = "#ff3535"
GREEN  = "#39d353"
WHITE  = "#eef2ff"
MUTED  = "#3d5070"
DIM    = "#1e2d45"

LABEL_COLOR = {"comply": CYAN, "hedge": AMBER, "refuse": RED,
               "partial": "#cc88ff", "evade": "#aaaaaa"}

TIER_ORDER = ["benign_bio", "cannabis", "psilocybin", "dual_use", "hazard_adjacent"]
TIER_LABEL = {
    "benign_bio":      "Benign Biology",
    "cannabis":        "Cannabis  (Sched. I)",
    "psilocybin":      "Psilocybin  (Sched. I)",
    "dual_use":        "Dual-Use Research",
    "hazard_adjacent": "Hazard-Adjacent",
}
TIER_ACCENT = {
    "benign_bio":      CYAN,
    "cannabis":        "#44cc55",
    "psilocybin":      "#cc44bb",
    "dual_use":        AMBER,
    "hazard_adjacent": RED,
}

# ── Load data ──────────────────────────────────────────────────────────────────
with open(DATA) as f:
    raw = json.load(f)["records"]
records = [r for r in raw if r.get("D_200") is not None]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9.3), facecolor=BG, dpi=150)
gs  = GridSpec(1, 2, width_ratios=[1, 1.55],
               left=0.02, right=0.98, top=0.97, bottom=0.06, wspace=0.04)

ax_l = fig.add_subplot(gs[0])  # left  – concept panel
ax_r = fig.add_subplot(gs[1])  # right – data panel

for ax in (ax_l, ax_r):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — concept visualization
# ══════════════════════════════════════════════════════════════════════════════
ax_l.set_xlim(0, 1)
ax_l.set_ylim(0, 1)
ax_l.axis("off")

rng = np.random.default_rng(42)

# ── Background gradient: surface (top) = deep blue, deep (bottom) = dark red ──
from matplotlib.patches import Rectangle
for yi in np.linspace(0, 1, 80):
    t    = 1 - yi          # t=0 top (surface), t=1 bottom (deep)
    r_c  = t * 0.25
    g_c  = 0.04
    b_c  = 0.18 * (1 - t) + 0.04
    rect = Rectangle((0, yi - 0.007), 1, 0.022,
                      facecolor=(r_c, g_c, b_c), edgecolor="none",
                      transform=ax_l.transAxes, zorder=0)
    ax_l.add_patch(rect)

# ── Neural net nodes + edges ───────────────────────────────────────────────────
n_nodes = 90
nx = rng.uniform(0.05, 0.95, n_nodes)
ny = rng.uniform(0.04, 0.96, n_nodes)

for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        d = np.hypot(nx[i]-nx[j], ny[i]-ny[j])
        if d < 0.15:
            t_ij = 1 - (ny[i]+ny[j]) / 2    # depth proxy
            alpha = (0.15 - d) / 0.15 * 0.22
            col   = (0.9 * t_ij, 0.05, 0.8 * (1 - t_ij))
            ax_l.plot([nx[i], nx[j]], [ny[i], ny[j]],
                      color=col, alpha=alpha, lw=0.5, zorder=1)

for i in range(n_nodes):
    t = 1 - ny[i]
    r = min(0.1 + 0.9 * t, 1.0)
    g = max(0.7 * (1 - t), 0.0)
    b = max(0.9 * (1 - t) + 0.1 * t, 0.0)
    sz = 18 + 45 * t * rng.uniform(0.6, 1.4)
    ax_l.scatter(nx[i], ny[i], s=sz, color=(r, g, b, 0.7),
                 zorder=2, edgecolors="none")

# ── Divider line with glow ────────────────────────────────────────────────────
div_y = 0.535
for lw, alpha in [(10, 0.05), (4, 0.12), (1.5, 0.7)]:
    ax_l.axhline(div_y, xmin=0.01, xmax=0.99,
                 color=WHITE, lw=lw, alpha=alpha, zorder=5)

ax_l.text(0.50, div_y + 0.030, "▲  SURFACE BEHAVIOR",
          ha="center", va="bottom", fontsize=8, color=CYAN,
          fontweight="bold", zorder=6,
          path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
ax_l.text(0.50, div_y - 0.030, "▼  INTERNAL ACTIVATIONS",
          ha="center", va="top",   fontsize=8, color=RED,
          fontweight="bold", zorder=6,
          path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

# ── "REFUSE" badge in the surface zone ───────────────────────────────────────
refuse_box = mpatches.FancyBboxPatch(
    (0.28, 0.68), 0.44, 0.10,
    boxstyle="round,pad=0.015",
    facecolor=GREEN, edgecolor="#00ff44", lw=1.2,
    transform=ax_l.transAxes, zorder=7, alpha=0.85
)
ax_l.add_patch(refuse_box)
ax_l.text(0.50, 0.730, "✓  REFUSE",
          ha="center", va="center", fontsize=15, fontweight="bold",
          color=BG, zorder=8,
          path_effects=[pe.withStroke(linewidth=2, foreground="#005500")])

# ── "HAZARD FEATURES ACTIVE" badge in the deep zone ─────────────────────────
deep_box = mpatches.FancyBboxPatch(
    (0.10, 0.28), 0.80, 0.10,
    boxstyle="round,pad=0.015",
    facecolor="#3a0808", edgecolor=RED, lw=1.2,
    transform=ax_l.transAxes, zorder=7, alpha=0.92
)
ax_l.add_patch(deep_box)
ax_l.text(0.50, 0.330, "⚠  HAZARD FEATURES ACTIVE",
          ha="center", va="center", fontsize=10, fontweight="bold",
          color=RED, zorder=8,
          path_effects=[pe.withStroke(linewidth=3, foreground="#1a0000")])

# ── Refusal depth meter ───────────────────────────────────────────────────────
mx, my_top, my_bot = 0.90, div_y - 0.02, 0.175
ax_l.annotate("", xy=(mx, my_bot), xytext=(mx, my_top),
              arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.0,
                              mutation_scale=16), zorder=9)
ax_l.text(mx + 0.02, (my_top + my_bot) / 2, "Refusal\nDepth",
          ha="left", va="center", fontsize=7.5, color=AMBER,
          fontweight="bold", zorder=9,
          path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

# ── Title ─────────────────────────────────────────────────────────────────────
ax_l.text(0.50, 0.975, "BioRefusalAudit",
          ha="center", va="top", fontsize=26, fontweight="bold", color=WHITE,
          zorder=12,
          path_effects=[pe.withStroke(linewidth=7, foreground=BG)])
ax_l.text(0.50, 0.915, "Auditing Biosecurity Refusal Depth",
          ha="center", va="top", fontsize=9.5, color=CYAN, alpha=0.95, zorder=12,
          path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
ax_l.text(0.50, 0.882, "via Sparse Autoencoder Feature Probing",
          ha="center", va="top", fontsize=9.5, color=CYAN, alpha=0.95, zorder=12,
          path_effects=[pe.withStroke(linewidth=3, foreground=BG)])

# ── Key findings boxes ────────────────────────────────────────────────────────
findings = [
    (AMBER,  "0 %",  "hazard-adjacent prompts refused  (80-tok, Gemma 2)"),
    (RED,    "33 %", "psilocybin refused vs  0 % hazardous biology"),
    (CYAN,   "D",    "cleanly separates comply vs refuse, zero overlap"),
    (GREEN,  "65→0", "refusals: format-gate removes all  (Gemma 4)"),
]
fy_start = 0.188
box_h   = 0.068
gap     = 0.076
for i, (col, val, desc) in enumerate(findings):
    fy = fy_start - i * gap
    if fy - box_h / 2 < 0.015:
        break
    rect = mpatches.FancyBboxPatch(
        (0.03, fy - box_h / 2 - 0.002), 0.94, box_h,
        boxstyle="round,pad=0.006",
        facecolor=DIM, edgecolor=col, lw=0.8,
        transform=ax_l.transAxes, zorder=10, alpha=0.9
    )
    ax_l.add_patch(rect)
    ax_l.text(0.065, fy, val,
              ha="left", va="center", fontsize=12, fontweight="bold",
              color=col, zorder=11, transform=ax_l.transAxes,
              path_effects=[pe.withStroke(linewidth=2, foreground=DIM)])
    ax_l.text(0.285, fy, desc,
              ha="left", va="center", fontsize=6.5, color=WHITE,
              alpha=0.90, zorder=11, transform=ax_l.transAxes)

ax_l.text(0.50, 0.010,
          "AIxBio Hackathon 2026  ·  Track 3: Biosecurity Tools  ·  Fourth Eon Bio",
          ha="center", va="bottom", fontsize=5.8, color=MUTED,
          transform=ax_l.transAxes, zorder=12)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — D-value distribution + scatter by tier
# ══════════════════════════════════════════════════════════════════════════════
ax = ax_r
ax.set_facecolor(PANEL)

tier_idx = {t: i for i, t in enumerate(TIER_ORDER)}

# Gather per-tier data
tier_data = {t: {"d": [], "labels": []} for t in TIER_ORDER}
for r in records:
    t = r.get("tier", "")
    if t in tier_data:
        tier_data[t]["d"].append(r["D_200"])
        tier_data[t]["labels"].append(r.get("label_200", "comply"))

# ── KDE violin-style distributions ───────────────────────────────────────────
kde_width = 0.38
x_grid    = np.linspace(0.0, 1.0, 300)

for i, tier in enumerate(TIER_ORDER):
    vals = np.array(tier_data[tier]["d"])
    if len(vals) < 3:
        continue
    try:
        kde = gaussian_kde(vals, bw_method=0.25)
        dens = kde(x_grid)
        dens = dens / dens.max() * kde_width
    except Exception:
        continue
    col = TIER_ACCENT[tier]
    ax.fill_between(x_grid, i - dens, i + dens,
                    color=col, alpha=0.18, zorder=2)
    ax.plot(x_grid, i + dens, color=col, lw=0.8, alpha=0.5, zorder=3)
    ax.plot(x_grid, i - dens, color=col, lw=0.8, alpha=0.5, zorder=3)

# ── Jittered scatter ──────────────────────────────────────────────────────────
for tier in TIER_ORDER:
    i = tier_idx[tier]
    for d, lbl in zip(tier_data[tier]["d"], tier_data[tier]["labels"]):
        col    = LABEL_COLOR.get(lbl, WHITE)
        jitter = rng.uniform(-0.22, 0.22)
        ax.scatter(d, i + jitter, s=20, color=col,
                   alpha=0.72, edgecolors="none", zorder=4)

# ── Mean marker ───────────────────────────────────────────────────────────────
for i, tier in enumerate(TIER_ORDER):
    vals = tier_data[tier]["d"]
    if not vals:
        continue
    mean_d = np.mean(vals)
    ax.plot([mean_d], [i], marker="D", ms=5.5, color=WHITE,
            zorder=6, markeredgecolor=BG, markeredgewidth=0.6)
    ax.text(mean_d + 0.022, i + 0.30,
            f"μ = {mean_d:.2f}",
            ha="left", va="top", fontsize=6.5, color=WHITE, alpha=0.75, zorder=6)

# ── Tier color strip on left ──────────────────────────────────────────────────
for i, tier in enumerate(TIER_ORDER):
    col = TIER_ACCENT[tier]
    rect = mpatches.FancyBboxPatch(
        (-0.055, i - 0.43), 0.030, 0.86,
        boxstyle="round,pad=0.002",
        facecolor=col, alpha=0.75,
        transform=ax.transData, zorder=5,
    )
    ax.add_patch(rect)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlim(-0.06, 1.06)
ax.set_ylim(-0.58, len(TIER_ORDER) - 0.42)
ax.set_yticks(range(len(TIER_ORDER)))
ax.set_yticklabels([TIER_LABEL[t] for t in TIER_ORDER],
                   fontsize=9, color=WHITE)
ax.set_xlabel("Divergence Score  D  (internal feature state ↔ surface behavior)",
              fontsize=9, color=MUTED, labelpad=8)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.0"],
                   fontsize=7.5, color=MUTED)
ax.tick_params(axis="y", length=0, pad=8)
ax.tick_params(axis="x", color=MUTED, length=3)

# Grid
for xv in [0.25, 0.5, 0.75]:
    ax.axvline(xv, color=DIM, lw=0.7, zorder=1)
ax.axvline(0.0, color=MUTED, lw=0.5, zorder=1)
for yi in range(len(TIER_ORDER)):
    ax.axhline(yi, color=DIM, lw=0.4, alpha=0.6, zorder=1)

# ── Key annotation ────────────────────────────────────────────────────────────
# psilocybin is index 2, hazard_adjacent is 4
ax.annotate(
    "Refusal circuit fires on\ncultural salience, not CBRN risk\n(psilocybin 33% refused, hazard 0%)",
    xy=(0.03, tier_idx["psilocybin"]),
    xytext=(0.42, 1.2),
    fontsize=7.2, color=AMBER,
    arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.0,
                    connectionstyle="arc3,rad=0.25"),
    bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL, edgecolor=AMBER, lw=0.9),
    zorder=8,
)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "SAE-Derived Divergence Score by Prompt Tier   (Gemma 2 2B-IT, n = 75)",
    fontsize=9.5, color=WHITE, pad=12,
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=CYAN,  label="Comply"),
    mpatches.Patch(facecolor=AMBER, label="Hedge"),
    mpatches.Patch(facecolor=RED,   label="Refuse"),
    Line2D([0], [0], marker="D", color=WHITE, ms=5, lw=0,
           markeredgecolor=BG, label="Mean D"),
]
leg = ax.legend(
    handles=legend_elements, loc="upper right",
    framealpha=0.25, edgecolor=MUTED, fontsize=8,
    labelcolor=WHITE, facecolor=PANEL,
    borderpad=0.7, handlelength=1.2,
)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Written: {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
