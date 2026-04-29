"""
Figure 4: Predictability spectrum and mediation b-path identity.

Panel A: predicted vs chronological age across three organ models.
Panel B: mediation framework schematic (A -> B -> Y).
Panel C: gap-disease coefficient vs mediation b-path coefficient.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# ---- Inputs ----
AGE_DATA_PATH = "data/UKB_aging_model_age_prediction.csv"
MED_DATA_PATH = "data/mediation_atlas_results.csv"
OUTPUT_PNG    = "fig4.png"
OUTPUT_PDF    = "fig4.pdf"
DPI           = 300

# ---- Style ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# ---- Panel A: three organ models ----
TISSUES_3 = [
    ("Organismal", r"$r = 0.90$", "#2166ac"),
    ("Brain",      r"$r = 0.72$", "#762a83"),
    ("Kidney",     r"$r = 0.16$", "#d6604d"),
]

# ---- Panel C: 12 organs and 13 diseases ----
TISSUES_12 = sorted(["Immune", "Lung", "Heart", "Intestine", "Adipose",
                     "Pancreas", "Artery", "Liver", "Brain", "Muscle",
                     "Kidney", "Organismal"])

DISEASES_13 = [
    "chronic_kidney_disease", "type2_diabetes", "osteoarthritis",
    "ischemic_heart_disease", "chronic_liver_disease", "all_cause_dementia",
    "atrial_fibrillation_or_flutter", "cerebrovascular_disease",
    "heart_failure", "rheumatoid_arthritis", "osteoporosis",
    "vascular_dementia", "alzheimer_disease",
]

_PALETTE_12 = [
    "#1f77b4", "#d62728", "#9467bd", "#2ca02c", "#ff7f0e", "#8c564b",
    "#17becf", "#bcbd22", "#e377c2", "#7f7f7f", "#393b79", "#f7b6d2",
]
ORGAN_COLORS = dict(zip(TISSUES_12, _PALETTE_12))


# ---- Load ----
df_age = pd.read_csv(AGE_DATA_PATH)
df_med = pd.read_csv(MED_DATA_PATH)
rng    = np.random.default_rng(42)


def density(x, y, n_s=4000):
    pts = np.vstack([x, y])
    idx = rng.choice(len(x), min(len(x), n_s), replace=False)
    return gaussian_kde(pts[:, idx])(pts)


# ---- Figure layout ----
fig = plt.figure(figsize=(12, 9.5))
gs_outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1],
                             hspace=0.55, top=0.94, bottom=0.08,
                             left=0.07, right=0.97)
gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0],
                                          wspace=0.32)
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[1],
                                          wspace=0.38, width_ratios=[1, 1.3])
ax_a = [fig.add_subplot(gs_top[i]) for i in range(3)]
ax_b = fig.add_subplot(gs_bot[0])
ax_c = fig.add_subplot(gs_bot[1])

XLIM, YLIM = [37, 73], [34, 78]
XTICK = YTICK = [40, 50, 60, 70]


# ---- Panel A ----
for ax, (tissue, r_label, color) in zip(ax_a, TISSUES_3):
    sub  = df_age[df_age["tissue"] == tissue].copy()
    age  = sub["Age"].values.astype(float)
    pred = sub["pred_Age"].values

    age_jit = age + rng.uniform(-0.45, 0.45, len(age))
    slope, intercept, _, _, _ = stats.linregress(age, pred)
    age_grid = np.linspace(XLIM[0], XLIM[1], 300)

    dens = density(age_jit, pred)
    so   = dens.argsort()
    cmap_org = LinearSegmentedColormap.from_list("org", ["#f0f0f0", color])
    ax.scatter(age_jit[so], pred[so], c=dens[so], cmap=cmap_org,
               s=3, alpha=0.22, edgecolor="none",
               rasterized=True, zorder=2)

    ax.plot(age_grid, slope * age_grid + intercept, color=color,
            lw=2.0, zorder=5)
    ax.plot(XLIM, XLIM, color="#bbb", lw=1.0, ls="--", zorder=1)

    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_xticks(XTICK); ax.set_yticks(YTICK)
    ax.set_xlabel("Chronological age (years)", fontsize=9)
    ax.set_ylabel("Predicted age (years)", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.set_title(tissue, fontsize=9, fontweight="bold", color=color, pad=5)
    ax.text(0.97, 0.03, r_label, transform=ax.transAxes,
            fontsize=8.5, ha="right", va="bottom",
            color=color, fontweight="bold")


# ---- Panel B: mediation diagram ----
ax = ax_b
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.axis("off")

BOX_W, BOX_H = 3.6, 1.0


def draw_box(ax, cx, cy, text, fc, ec):
    box = FancyBboxPatch((cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
                         boxstyle="round,pad=0.15",
                         facecolor=fc, edgecolor=ec, lw=1.2, zorder=5)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=8.5, fontweight="bold", zorder=6)


def draw_arrow(ax, x1, y1, x2, y2, label, color,
               above=True, curve=0, ls="-"):
    style = f"arc3,rad={curve}"
    arrow_kw = dict(arrowstyle="->", color=color, lw=1.6,
                    connectionstyle=style, mutation_scale=10)
    if ls == "--":
        arrow_kw["linestyle"] = "dashed"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=arrow_kw, zorder=4)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dy = 0.32 if above else -0.32
        ax.text(mx, my + dy, label, ha="center", va="center",
                fontsize=9.5, color=color, style="italic",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=1.5))


nodes = {"A": (1.9, 3.2), "B": (5.0, 5.6), "Y": (8.1, 3.2)}
node_specs = [
    ("A", "#2166ac", "Chronological age (A)", "#e8f4f8"),
    ("B", "#762a83", "Biological age biomarker (B)", "#f3eef8"),
    ("Y", "#d6604d", "Disease outcome (Y)", "#fef0ec"),
]
for key, ec, text, fc in node_specs:
    cx, cy = nodes[key]
    draw_box(ax, cx, cy, text, fc=fc, ec=ec)

Ax, Ay = nodes["A"]; Bx, By = nodes["B"]; Yx, Yy = nodes["Y"]
draw_arrow(ax, Ax + BOX_W / 2, Ay, Bx - BOX_W / 2, By,
           "a", "#762a83", above=True)
draw_arrow(ax, Bx + BOX_W / 2, By, Yx - BOX_W / 2, Yy,
           "b", "#d6604d", above=True)
draw_arrow(ax, Ax + BOX_W / 2, Ay, Yx - BOX_W / 2, Yy,
           "c'", "#555", above=False, curve=-0.20, ls="--")

ax.text(5.0, 7.05, "Indirect effect (a × b)",
        ha="center", va="center", fontsize=9,
        color="#9e0020", fontweight="bold", style="italic")
ax.text(5.0, 2.45, "Direct effect",
        ha="center", va="center", fontsize=9,
        color="#444", fontweight="bold", style="italic")
ax.set_title("Mediation framework of aging clock analyses",
             fontsize=9, fontweight="bold", pad=6)


# ---- Panel C: identity scatter ----
ax = ax_c
df_sub = df_med[
    df_med["Tissue"].isin(TISSUES_12) &
    df_med["Disease"].isin(DISEASES_13)
].copy()

x_vals = df_sub["beta_residual"].values
y_vals = df_sub["bioage_to_disease_effect"].values
colors = [ORGAN_COLORS.get(t, "#888") for t in df_sub["Tissue"].values]

ax.scatter(x_vals, y_vals, c=colors, s=18, alpha=0.80,
           edgecolor="white", linewidth=0.4,
           zorder=3, rasterized=True)

pad = 0.004
lim_lo = min(x_vals.min() - pad, -pad)
lim_hi = x_vals.max() + pad
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
        color="#ddd", lw=1.3, ls="--", zorder=4)

ax.set_xlim([lim_lo, lim_hi])
ax.set_ylim([lim_lo, lim_hi])
ax.set_aspect("equal")
ticks = np.arange(0, lim_hi + 0.01, 0.02)
ax.set_xticks(ticks); ax.set_yticks(ticks)

ax.set_xlabel("Δres → disease coefficient", fontsize=9)
ax.set_ylabel("Mediation b-path coefficient (b)", fontsize=9)
ax.tick_params(labelsize=8.5)

legend_els = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=ORGAN_COLORS[t], markersize=6, label=t)
    for t in TISSUES_12
]
ax.legend(handles=legend_els, fontsize=8, frameon=False,
          loc="lower right", ncol=2, handlelength=0.8,
          labelspacing=0.3, columnspacing=0.5)


# ---- Panel labels ----
for label, ax_l, xoff, yoff in [
    ("A", ax_a[0], -0.18, 1.08),
    ("B", ax_b,    -0.08, 1.08),
    ("C", ax_c,    -0.22, 1.08),
]:
    ax_l.text(xoff, yoff, label, transform=ax_l.transAxes,
              fontsize=12, fontweight="bold", va="top", ha="left")

fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUTPUT_PNG}, {OUTPUT_PDF}")
