"""
Figure 2: Sex-specific bias in age gaps and personalized calibration.

Panel A: biomarker space with pooled and sex-specific calibration lines.
Panel B: gap distribution before sex personalization.
Panel C: gap distribution after sex personalization.
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
from matplotlib.colors import LinearSegmentedColormap

# ---- Inputs ----
DATA_PATH  = "data/UKB_aging_model_age_prediction.csv"
TISSUE     = "Artery"
OUTPUT_PNG = "fig2.png"
OUTPUT_PDF = "fig2.pdf"
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

XLIM      = [37, 73]
YLIM_TOP  = [37, 73]
YLIM_GAP  = [-9, 9]
XTICK     = [40, 50, 60, 70]
YTICK_TOP = [40, 50, 60, 70]
YTICK_GAP = [-8, -4, 0, 4, 8]

S_SIZE   = 3
ALPHA    = 0.22
FS_LABEL = 9
FS_TICK  = 8.5

CF      = "#9e0020"
CM      = "#003d8f"
CMAP_F  = LinearSegmentedColormap.from_list("f", ["#f7c5cc", "#9e0020"])
CMAP_M  = LinearSegmentedColormap.from_list("m", ["#b8d4f0", "#003d8f"])

# ---- Load ----
df = pd.read_csv(DATA_PATH)
df = df[df["split"] == "test"]
sub = df[df["tissue"] == TISSUE].reset_index(drop=True)

age  = sub["Age"].values.astype(float)
pred = sub["pred_Age"].values
sex  = sub["Sex"].values
f_mask = sex == 0
m_mask = sex == 1

rng      = np.random.default_rng(42)
age_jit  = age + rng.uniform(-0.45, 0.45, len(age))
age_grid = np.linspace(XLIM[0], XLIM[1], 400)

# ---- Calibration regressions ----
sl_pool, ic_pool, r_pool, _, _ = stats.linregress(age, pred)

# Sex-covariate model: pred = b0 + b_age * age + b_sex * sex
X = np.column_stack([np.ones(len(age)), age, sex])
beta, _, _, _ = np.linalg.lstsq(X, pred, rcond=None)
b0, b_age, b_sex = beta
line_f = b0 + b_age * age_grid
line_m = b0 + b_sex + b_age * age_grid

_, _, r_f, _, _ = stats.linregress(age[f_mask], pred[f_mask])
_, _, r_m, _, _ = stats.linregress(age[m_mask], pred[m_mask])

# ---- Gaps ----
gap_before = pred - (sl_pool * age + ic_pool)
gap_after  = pred - (b0 + b_age * age + b_sex * sex)

mfb = gap_before[f_mask].mean()
mmb = gap_before[m_mask].mean()
mfa = gap_after[f_mask].mean()
mma = gap_after[m_mask].mean()


def kde(x, y, max_pts=4000):
    pts = np.vstack([x, y])
    idx = rng.choice(len(x), min(len(x), max_pts), replace=False)
    return gaussian_kde(pts[:, idx])(pts)


d_f   = kde(age_jit[f_mask], pred[f_mask])
d_m   = kde(age_jit[m_mask], pred[m_mask])
d_gbf = kde(age_jit[f_mask], gap_before[f_mask])
d_gbm = kde(age_jit[m_mask], gap_before[m_mask])
d_gaf = kde(age_jit[f_mask], gap_after[f_mask])
d_gam = kde(age_jit[m_mask], gap_after[m_mask])


def scatter_sex(ax, xf, yf, df_, xm, ym, dm):
    sf = df_.argsort(); sm = dm.argsort()
    ax.scatter(xf[sf], yf[sf], c=df_[sf], cmap=CMAP_F,
               s=S_SIZE, alpha=ALPHA, edgecolor="none",
               rasterized=True, zorder=2)
    ax.scatter(xm[sm], ym[sm], c=dm[sm], cmap=CMAP_M,
               s=S_SIZE, alpha=ALPHA, edgecolor="none",
               rasterized=True, zorder=2)


def style_ax(ax, xlim, ylim, xtick, ytick, xlabel, ylabel):
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks(xtick); ax.set_yticks(ytick)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)


# ---- Figure ----
fig = plt.figure(figsize=(15, 5.2))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                        top=0.88, bottom=0.13, left=0.07, right=0.97)
ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

# ---- Panel A ----
ax = ax_a
scatter_sex(ax, age_jit[f_mask], pred[f_mask], d_f,
                age_jit[m_mask], pred[m_mask], d_m)
ax.plot(XLIM, XLIM, color="#bbb", lw=1.0, ls="--", zorder=1)
ax.plot(age_grid, sl_pool * age_grid + ic_pool, color="#333", lw=1.5, zorder=5)
ax.plot(age_grid, line_f, color=CF, lw=1.5, zorder=6)
ax.plot(age_grid, line_m, color=CM, lw=1.5, zorder=6)

ax.text(0.97, 0.14, f"r = {r_pool:.2f}", transform=ax.transAxes,
        fontsize=8, ha="right", va="bottom", color="black")
ax.text(0.97, 0.09, f"r = {r_f:.2f}", transform=ax.transAxes,
        fontsize=8, ha="right", va="bottom", color=CF)
ax.text(0.97, 0.04, f"r = {r_m:.2f}", transform=ax.transAxes,
        fontsize=8, ha="right", va="bottom", color=CM)

style_ax(ax, XLIM, YLIM_TOP, XTICK, YTICK_TOP,
         "Chronological age (years)", "Predicted age (years)")
ax.set_title("Artery aging", fontsize=9, fontweight="bold", pad=5)

leg_lines = [
    Line2D([0], [0], color="black", lw=1.5, label="Pooled"),
    Line2D([0], [0], color=CF,      lw=1.5, label="Female"),
    Line2D([0], [0], color=CM,      lw=1.5, label="Male"),
]
ax.legend(handles=leg_lines, fontsize=7.5, frameon=False,
          loc="upper left", handlelength=1.5, labelspacing=0.35)

# ---- Panel B: before personalization ----
ax = ax_b
scatter_sex(ax, age_jit[f_mask], gap_before[f_mask], d_gbf,
                age_jit[m_mask], gap_before[m_mask], d_gbm)
ax.axhline(0,   color="#888", lw=0.9, ls="--", zorder=1)
ax.axhline(mfb, color=CF, lw=1.8, zorder=4, alpha=0.9)
ax.axhline(mmb, color=CM, lw=1.8, zorder=4, alpha=0.9)

ax.text(XLIM[1] - 0.5, mfb + 0.35, f"Female: {mfb:+.1f} y",
        color=CF, fontsize=7.5, ha="right", va="bottom", fontweight="bold")
ax.text(XLIM[1] - 0.5, mmb - 0.35, f"Male: {mmb:+.1f} y",
        color=CM, fontsize=7.5, ha="right", va="top", fontweight="bold")

style_ax(ax, XLIM, YLIM_GAP, XTICK, YTICK_GAP,
         "Chronological age (years)", "Age gap (years)")
ax.set_title("Without sex personalization", fontsize=9, fontweight="bold", pad=5)

# ---- Panel C: after personalization ----
ax = ax_c
scatter_sex(ax, age_jit[f_mask], gap_after[f_mask], d_gaf,
                age_jit[m_mask], gap_after[m_mask], d_gam)
ax.axhline(0,   color="#888", lw=0.9, ls="--", zorder=1)
ax.axhline(mfa, color=CF, lw=1.8, zorder=4, alpha=0.9)
ax.axhline(mma, color=CM, lw=1.8, zorder=4, alpha=0.9)

ax.text(XLIM[1] - 0.5, mfa + 0.35, f"Female: {abs(mfa):.1f} y",
        color=CF, fontsize=7.5, ha="right", va="bottom", fontweight="bold")
ax.text(XLIM[1] - 0.5, mma - 0.35, f"Male: {abs(mma):.1f} y",
        color=CM, fontsize=7.5, ha="right", va="top", fontweight="bold")

style_ax(ax, XLIM, YLIM_GAP, XTICK, YTICK_GAP,
         "Chronological age (years)", "Age gap (years)")
ax.set_title("With sex personalization", fontsize=9, fontweight="bold", pad=5)

# ---- Panel labels ----
for label, ax_l in [("A", ax_a), ("B", ax_b), ("C", ax_c)]:
    ax_l.text(-0.10, 1.06, label, transform=ax_l.transAxes,
              fontsize=12, fontweight="bold", va="top", ha="left")

legend_els = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=CF,
           markersize=7, label="Female"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=CM,
           markersize=7, label="Male"),
]
fig.legend(handles=legend_els, loc="upper center", ncol=2, fontsize=8,
           frameon=False, bbox_to_anchor=(0.54, 1.01),
           handletextpad=0.4, columnspacing=1.2)

fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUTPUT_PNG}, {OUTPUT_PDF}")
