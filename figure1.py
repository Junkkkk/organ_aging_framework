"""
Figure 1: Calibration regression and gap definitions.

Panel A: naive gap (B - A), correlated with age by construction.
Panel B: residual gap (B - regression line).
Panel C: calibrated gap (horizontal distance to regression line).
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---- Inputs ----
DATA_PATH  = "data/UKB_aging_model_age_prediction.csv"
TISSUE     = "Brain"
OUTPUT_PNG = "fig1.png"
OUTPUT_PDF = "fig1.pdf"
DPI        = 300

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

XLIM, YLIM = [37, 73], [34, 78]
XTICK = YTICK = [40, 50, 60, 70]

C_REG     = "#333333"
C_POINT   = "#c62828"
C_GAP     = "#c62828"
C_BIOAGE  = "#1565c0"
FS_LABEL  = 9
FS_TICK   = 8.5
FS_ANNOT  = 8

# ---- Load ----
df = pd.read_csv(DATA_PATH)
df = df[df["split"] == "test"]
sub = df[df["tissue"] == TISSUE].reset_index(drop=True)

age  = sub["Age"].values.astype(float)
pred = sub["pred_Age"].values
rng  = np.random.default_rng(42)

slope, intercept, r, _, _ = stats.linregress(age, pred)
age_grid = np.linspace(XLIM[0], XLIM[1], 400)
reg_line = slope * age_grid + intercept

# ---- Pick example individual near (45, 60) for the illustration ----
mask = (age >= 42) & (age <= 46)
cands = sub[mask].copy()
cands = cands[(cands["pred_Age"] >= 57) & (cands["pred_Age"] <= 65)]
cands["dist"] = abs(cands["pred_Age"] - 60)
ex_idx = cands.nsmallest(1, "dist").index[0]

Px   = float(sub.loc[ex_idx, "Age"])
Py   = float(sub.loc[ex_idx, "pred_Age"])
Preg = slope * Px + intercept
bio_age = (Py - intercept) / slope

naive_gap = Py - Px
res_gap   = Py - Preg
cal_gap   = bio_age - Px

# ---- KDE for background scatter ----
age_jit = age + rng.uniform(-0.45, 0.45, len(age))
sub_idx = rng.choice(len(age), min(len(age), 4000), replace=False)
dens = gaussian_kde(np.vstack([age_jit, pred])[:, sub_idx])(np.vstack([age_jit, pred]))
order = dens.argsort()

# ---- Figure ----
fig = plt.figure(figsize=(15, 5.2))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                        top=0.88, bottom=0.13, left=0.07, right=0.97)
ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))


def draw_scatter(ax):
    ax.scatter(age_jit[order], pred[order], c=dens[order], cmap="Greys",
               s=3, alpha=0.22, edgecolor="none", rasterized=True, zorder=1,
               vmin=dens.min(), vmax=dens.max() * 0.7)


def style_ax(ax, ylabel="Predicted age (years)"):
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_xticks(XTICK); ax.set_yticks(YTICK)
    ax.set_xlabel("Chronological age (years)", fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)


def draw_example(ax):
    ax.scatter([Px], [Py], s=80, color=C_POINT,
               edgecolor="white", lw=0.8, zorder=8)


# ---- Panel A: naive gap ----
ax = ax_a
draw_scatter(ax)
ax.plot(XLIM, XLIM, color="#bbb", lw=1.0, ls="--", zorder=2)
ax.plot([Px, Px], [Px, Py], color=C_GAP, lw=2.0, zorder=6, solid_capstyle="round")
ax.scatter([Px], [Px], s=40, color="#888", edgecolor="white", lw=0.5, zorder=7)
draw_example(ax)
ax.plot([XLIM[0], Px], [Py, Py], color=C_POINT, lw=0.8, ls=":", alpha=0.7, zorder=4)
ax.plot([Px, Px], [YLIM[0], Px], color=C_POINT, lw=0.8, ls=":", alpha=0.7, zorder=4)

ax.annotate(f"Naive gap = {naive_gap:.1f} y",
            xy=(Px + 0.5, (Px + Py) / 2),
            xytext=(Px + 7, (Px + Py) / 2),
            fontsize=FS_ANNOT, ha="left", va="center", color=C_GAP,
            arrowprops=dict(arrowstyle="->", color=C_GAP, lw=1.0),
            bbox=dict(facecolor="white", edgecolor=C_GAP,
                      boxstyle="round,pad=0.3", lw=0.8, alpha=0.9))

ax.annotate(f"Age = {Px:.0f}",
            xy=(Px, YLIM[0] + 0.3),
            xytext=(Px + 5, YLIM[0] + 7),
            fontsize=7.5, ha="center", va="bottom", color=C_POINT,
            arrowprops=dict(arrowstyle="->", color=C_POINT, lw=0.9),
            bbox=dict(facecolor="white", edgecolor=C_POINT,
                      boxstyle="round,pad=0.25", lw=0.8, alpha=0.9))
style_ax(ax)
ax.set_title("Naive gap", fontsize=9, fontweight="bold", pad=4)

# ---- Panel B: residual gap ----
ax = ax_b
draw_scatter(ax)
ax.plot(XLIM, XLIM, color="#bbb", lw=1.0, ls="--", zorder=2)
ax.plot(age_grid, reg_line, color="#333", lw=1.5, zorder=5)
ax.plot([Px, Px], [Preg, Py], color=C_GAP, lw=2.0, zorder=6, solid_capstyle="round")
ax.scatter([Px], [Preg], s=40, color=C_REG, edgecolor="white", lw=0.5, zorder=7)
draw_example(ax)
ax.plot([XLIM[0], Px], [Py, Py], color=C_POINT, lw=0.8, ls=":", alpha=0.7, zorder=4)
ax.plot([Px, Px], [YLIM[0], Preg], color=C_POINT, lw=0.8, ls=":", alpha=0.7, zorder=4)

ax.annotate(f"Δres = {res_gap:.1f} y",
            xy=(Px, (Preg + Py) / 2),
            xytext=(Px + 9, (Preg + Py) / 2 + 5),
            fontsize=FS_ANNOT, ha="center", va="bottom", color=C_GAP,
            arrowprops=dict(arrowstyle="->", color=C_GAP, lw=1.0),
            bbox=dict(facecolor="white", edgecolor=C_GAP,
                      boxstyle="round,pad=0.3", lw=0.8, alpha=0.9))
style_ax(ax)
ax.set_title("Residual approach", fontsize=9, fontweight="bold", pad=4)

# ---- Panel C: calibrated gap ----
ax = ax_c
draw_scatter(ax)
ax.plot(XLIM, XLIM, color="#bbb", lw=1.0, ls="--", zorder=2)
ax.plot(age_grid, reg_line, color="#333", lw=1.5, zorder=5)
ax.plot([Px, bio_age], [Py, Py], color=C_GAP, lw=2.0, zorder=6, solid_capstyle="round")
ax.scatter([bio_age], [Py], s=40, color=C_REG, edgecolor="white", lw=0.5, zorder=7)
draw_example(ax)
ax.plot([Px, Px], [YLIM[0], Py], color=C_POINT, lw=0.8, ls=":", alpha=0.7, zorder=4)
ax.plot([bio_age, bio_age], [YLIM[0], Py], color=C_BIOAGE, lw=0.8, ls=":", alpha=0.7, zorder=4)

ax.annotate(f"Calibrated age = {bio_age:.1f}",
            xy=(bio_age, YLIM[0] + 0.3),
            xytext=(bio_age - 6, YLIM[0] + 7),
            fontsize=7.5, ha="center", va="bottom", color=C_BIOAGE,
            arrowprops=dict(arrowstyle="->", color=C_BIOAGE, lw=0.9),
            bbox=dict(facecolor="white", edgecolor=C_BIOAGE,
                      boxstyle="round,pad=0.25", lw=0.8, alpha=0.9))

mid_x = (Px + bio_age) / 2
ax.annotate(f"Δcal = {cal_gap:.1f} y",
            xy=(mid_x, Py + 0.3),
            xytext=(mid_x, Py + 6),
            fontsize=FS_ANNOT, ha="center", va="bottom", color=C_GAP,
            arrowprops=dict(arrowstyle="->", color=C_GAP, lw=1.0),
            bbox=dict(facecolor="white", edgecolor=C_GAP,
                      boxstyle="round,pad=0.3", lw=0.8, alpha=0.9))
style_ax(ax, ylabel="Predicted age / Calibrated age (years)")
ax.set_title("Calibrated approach", fontsize=9, fontweight="bold", pad=4)

# ---- Panel labels ----
for label, ax_l in [("A", ax_a), ("B", ax_b), ("C", ax_c)]:
    ax_l.text(-0.10, 1.06, label, transform=ax_l.transAxes,
              fontsize=12, fontweight="bold", va="top", ha="left")

fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUTPUT_PNG}, {OUTPUT_PDF}")
