"""
Figure 3: Prediction-interval stratification.

Panel A: PI-based vs SD-based cutoffs on a narrow training subsample.
Panel B: sex-personalized predictive distributions at ages 45, 55, 65.
Panel C: heart failure cumulative incidence by PI-stratified groups.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

# ---- Inputs ----
AGE_DATA_PATH = "data/UKB_aging_model_age_prediction.csv"
TTE_DATA_PATH = "data/tte_diseases.csv"
OUTPUT_PNG    = "fig3.png"
OUTPUT_PDF    = "fig3.pdf"
DPI           = 300

# ---- Style ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

TISSUE        = "Artery"
C_VAL         = 1.5            # PI/SD coefficient
T_MAX         = 15
FOCUS_AGES    = [45, 55, 65]
CLIP_SD       = 2.0
DWIDTH        = 2.0
PI_TICK_LW    = 1.5

SUB_N_TRAIN   = 120
SUB_AGE_RANGE = (54, 58)
SEED          = 42

CF       = "#9e0020"
CM       = "#003d8f"
CMAP_F   = LinearSegmentedColormap.from_list("f", ["#f7c5cc", "#9e0020"])
CMAP_M   = LinearSegmentedColormap.from_list("m", ["#b8d4f0", "#003d8f"])
COL_PI       = "#d6604d"
COL_SD       = "black"
COL_DISAGREE = "#2166ac"

C_OLD    = "#d6604d"
C_NORM   = "#888888"
C_YOUNG  = "#4393c3"


# ---- Load ----
print("Loading data...")
df_all = pd.read_csv(AGE_DATA_PATH)
df_tte = pd.read_csv(TTE_DATA_PATH, low_memory=False)

if "IID" in df_all.columns and "eid" not in df_all.columns:
    df_all["eid"] = df_all["IID"]

df_train = df_all[df_all["split"] == "train"].copy()
df_test  = df_all[df_all["split"] == "test"].copy()
print(f"  train n={len(df_train):,}, test n={len(df_test):,}")


# ---- Helpers ----
def fit_on(age_arr, pred_arr):
    n = len(age_arr)
    sl, ic, r, _, _ = stats.linregress(age_arr, pred_arr)
    sigma = (pred_arr - (sl * age_arr + ic)).std()
    abar  = age_arr.mean()
    ss    = np.sum((age_arr - abar) ** 2)
    return dict(slope=sl, intercept=ic, sigma=sigma,
                n=n, age_mean=abar, ss_age=ss, r=r)


def pi_halfwidth(age_arr, p):
    lev = (age_arr - p["age_mean"]) ** 2 / p["ss_age"]
    return C_VAL * p["sigma"] * np.sqrt(1 + 1 / p["n"] + lev)


def bands_on_grid(p, grid):
    line = p["slope"] * grid + p["intercept"]
    lev  = (grid - p["age_mean"]) ** 2 / p["ss_age"]
    pi_b = C_VAL * p["sigma"] * np.sqrt(1 + 1 / p["n"] + lev)
    sd_b = C_VAL * p["sigma"] * np.ones_like(grid)
    return line, pi_b, sd_b


def density(x, y, rng, n_s=4000):
    pts = np.vstack([x, y])
    idx = rng.choice(len(x), min(len(x), n_s), replace=False)
    return gaussian_kde(pts[:, idx])(pts)


def get_pi_groups(tissue, params):
    """Classify test individuals as old/normal/young using train-set PI.
    Sex-adjusted residuals are used for classification."""
    te = df_test[df_test["tissue"] == tissue].copy()
    merged = te.merge(
        df_tte[["eid", "age_at_baseline",
                "heart_failure_event", "heart_failure_time"]],
        on="eid", how="inner"
    ).dropna(subset=["heart_failure_event", "heart_failure_time"])

    age  = merged["Age"].values.astype(float)
    pred = merged["pred_Age"].values
    sex  = merged["Sex"].values
    Xm   = np.column_stack([np.ones(len(age)), age, sex])
    betam, _, _, _ = np.linalg.lstsq(Xm, pred, rcond=None)
    resid = pred - Xm @ betam
    hw    = pi_halfwidth(age, params)
    merged["pi_group"] = np.where(resid > hw, "old",
                          np.where(resid < -hw, "young", "normal"))
    return merged


def km_curve(time, event, t_max=15, n_pts=300):
    tg = np.linspace(0, t_max, n_pts)
    order = np.argsort(time)
    ts, es = time[order], event[order]
    S, cv, ptr = 1.0, 0.0, 0
    Sa, Va = [], []
    for t in tg:
        while ptr < len(ts) and ts[ptr] <= t:
            nr = len(ts) - ptr
            if nr > 0 and es[ptr] == 1:
                S *= (1 - 1 / nr)
                if nr > 1:
                    cv += 1 / (nr * (nr - 1))
            ptr += 1
        Sa.append(S); Va.append(cv)
    Sa, Va = np.array(Sa), np.array(Va)
    lo, hi = np.zeros_like(Sa), np.zeros_like(Sa)
    for i, (s, v) in enumerate(zip(Sa, Va)):
        if 0 < s < 1 and v > 0:
            u = np.exp(1.96 * np.sqrt(v) / abs(np.log(s)))
            lo[i], hi[i] = s ** u, s ** (1 / u)
        else:
            lo[i] = hi[i] = s
    return tg, 1 - Sa, 1 - hi, 1 - lo


# ---- Fit calibration models ----
print("Fitting calibration models...")

tr_full_artery = df_train[df_train["tissue"] == TISSUE]

# Panel A/B: narrow subsample
rng_sub = np.random.default_rng(SEED)
tr_narrow = tr_full_artery[(tr_full_artery["Age"] >= SUB_AGE_RANGE[0]) &
                           (tr_full_artery["Age"] <= SUB_AGE_RANGE[1])]
tr_sub = tr_narrow.iloc[rng_sub.choice(len(tr_narrow), SUB_N_TRAIN, replace=False)]

params_A = fit_on(tr_sub["Age"].values.astype(float),
                  tr_sub["pred_Age"].values)
print(f"  [A] narrow n={params_A['n']}, r={params_A['r']:.2f}")

# Panel B: sex-covariate model on the same subsample
a_sub   = tr_sub["Age"].values.astype(float)
p_sub   = tr_sub["pred_Age"].values
sex_sub = tr_sub["Sex"].values
nB = len(tr_sub)
XB = np.column_stack([np.ones(nB), a_sub, sex_sub])
betaB, _, _, _ = np.linalg.lstsq(XB, p_sub, rcond=None)
b0, b_age, b_sex = betaB
sigmaB  = (p_sub - XB @ betaB).std()
abarB   = a_sub.mean()
ssB     = np.sum((a_sub - abarB) ** 2)
print(f"  [B] sex-adjusted, b_sex={b_sex:.3f}")

# Panel C: full train set for PI classification
params_C = fit_on(tr_full_artery["Age"].values.astype(float),
                  tr_full_artery["pred_Age"].values)
print(f"  [C] full train n={params_C['n']:,}, r={params_C['r']:.2f}")


# ---- Test-set scatter prep ----
rng_plot = np.random.default_rng(SEED)
te_artery = df_test[df_test["tissue"] == TISSUE]

age_teA  = te_artery["Age"].values.astype(float)
pred_teA = te_artery["pred_Age"].values
jit_teA  = age_teA + rng_plot.uniform(-0.45, 0.45, len(age_teA))
resid_A  = pred_teA - (params_A["slope"] * age_teA + params_A["intercept"])
hw_A     = pi_halfwidth(age_teA, params_A)
sd_half_A = C_VAL * params_A["sigma"]
out_pi_A   = np.abs(resid_A) > hw_A
out_sd_A   = np.abs(resid_A) > sd_half_A
only_sd_A  = out_sd_A & ~out_pi_A
both_in_A  = ~out_sd_A & ~out_pi_A
both_out_A = out_sd_A & out_pi_A

age_teB  = te_artery["Age"].values.astype(float)
pred_teB = te_artery["pred_Age"].values
sex_teB  = te_artery["Sex"].values
f_mask   = sex_teB == 0
m_mask   = sex_teB == 1
jit_teB  = age_teB + rng_plot.uniform(-0.45, 0.45, len(age_teB))
print("  Computing density for Panel B...")
dens_f   = density(jit_teB[f_mask], pred_teB[f_mask], rng_plot)
dens_m   = density(jit_teB[m_mask], pred_teB[m_mask], rng_plot)

print("  Computing PI groups for Panel C...")
merged_C = get_pi_groups(TISSUE, params_C)


# ---- Figure ----
print("Rendering figure...")
fig = plt.figure(figsize=(17, 5.5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32,
                       top=0.90, bottom=0.13, left=0.05, right=0.98)
ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))


# ---- Panel A: PI vs SD ----
XLIM_A = [37, 73]
grid_A = np.linspace(XLIM_A[0], XLIM_A[1], 500)
line_A, pi_band_A, sd_band_A = bands_on_grid(params_A, grid_A)

ax_a.scatter(jit_teA[both_in_A], pred_teA[both_in_A],
             color="#bbb", s=3, alpha=0.15, edgecolor="none",
             zorder=2, rasterized=True)
ax_a.scatter(jit_teA[both_out_A], pred_teA[both_out_A],
             color="#222", s=4, alpha=0.28, edgecolor="none",
             zorder=3, rasterized=True)
ax_a.scatter(jit_teA[only_sd_A], pred_teA[only_sd_A],
             color=COL_DISAGREE, s=4.5, alpha=0.5, edgecolor="none",
             zorder=4, rasterized=True)

ax_a.plot(XLIM_A, XLIM_A, color="#bbb", lw=1.0, ls="--", zorder=1)
ax_a.plot(grid_A, line_A, color="black", lw=1.2, zorder=6)
ax_a.fill_between(grid_A, line_A - pi_band_A, line_A + pi_band_A,
                  color=COL_PI, alpha=0.10, zorder=3)
ax_a.plot(grid_A, line_A + pi_band_A, color=COL_PI, lw=1.2, ls="--",
          zorder=5, label="PI-based")
ax_a.plot(grid_A, line_A - pi_band_A, color=COL_PI, lw=1.2, ls="--", zorder=5)
ax_a.plot(grid_A, line_A + sd_band_A, color=COL_SD, lw=1.0, ls="--",
          zorder=5, label="SD-based")
ax_a.plot(grid_A, line_A - sd_band_A, color=COL_SD, lw=1.0, ls="--", zorder=5)

ax_a.set_xlim(XLIM_A); ax_a.set_ylim(XLIM_A)
ax_a.set_xticks([40, 50, 60, 70]); ax_a.set_yticks([40, 50, 60, 70])
ax_a.set_xlabel("Chronological age (years)", fontsize=9)
ax_a.set_ylabel("Predicted age (years)", fontsize=9)
ax_a.tick_params(labelsize=8.5)
ax_a.legend(frameon=False, loc="upper left", fontsize=8,
            handlelength=1.5, labelspacing=0.4)


# ---- Panel B: sex-personalized predictive distributions ----
XLIM_B = [37, 75]
YLIM_B = [36, 76]
grid_B = np.linspace(XLIM_B[0], XLIM_B[1], 400)
line_fB = b0 + b_age * grid_B
line_mB = b0 + b_age * grid_B + b_sex
lev_gB  = (grid_B - abarB) ** 2 / ssB
pi_band_B = C_VAL * sigmaB * np.sqrt(1 + 1 / nB + lev_gB)

for d, mask, cmap in [(dens_f, f_mask, CMAP_F), (dens_m, m_mask, CMAP_M)]:
    so = d.argsort()
    ax_b.scatter(jit_teB[mask][so], pred_teB[mask][so],
                 c=d[so], cmap=cmap, s=3, alpha=0.10,
                 edgecolor="none", rasterized=True, zorder=2)

ax_b.plot(XLIM_B, XLIM_B, color="#bbb", lw=1.0, ls="--", zorder=1)
ax_b.plot(grid_B, line_fB, color=CF, lw=1.3, zorder=6)
ax_b.plot(grid_B, line_mB, color=CM, lw=1.3, zorder=6)

for line_sex, color in [(line_fB, CF), (line_mB, CM)]:
    ax_b.fill_between(grid_B, line_sex - pi_band_B, line_sex + pi_band_B,
                      color=color, alpha=0.08, zorder=3)
    ax_b.plot(grid_B, line_sex + pi_band_B, color=color, lw=1.1,
              ls="--", alpha=0.8, zorder=5)
    ax_b.plot(grid_B, line_sex - pi_band_B, color=color, lw=1.1,
              ls="--", alpha=0.8, zorder=5)

# Bell curves at focus ages — female on the left, male on the right
for fa in FOCUS_AGES:
    lev_fa = (fa - abarB) ** 2 / ssB
    pi_fa  = C_VAL * sigmaB * np.sqrt(1 + 1 / nB + lev_fa)

    for s_val, color in [(0, CF), (1, CM)]:
        mu = b0 + b_age * fa + b_sex * s_val
        y_rng = np.linspace(mu - CLIP_SD * sigmaB, mu + CLIP_SD * sigmaB, 300)
        gauss = norm.pdf(y_rng, mu, sigmaB)
        g_sc  = gauss / norm.pdf(mu, mu, sigmaB) * DWIDTH

        if s_val == 0:
            x_curve   = fa - g_sc
            tick_xs   = lambda g: [fa - g, fa]
        else:
            x_curve   = fa + g_sc
            tick_xs   = lambda g: [fa, fa + g]

        ax_b.fill_betweenx(y_rng, fa, x_curve,
                           color=color, alpha=0.30, zorder=7)
        ax_b.plot(x_curve, y_rng, color=color, lw=1.0, alpha=0.85, zorder=8)

        for sign in [-1, 1]:
            y_pi = mu + sign * pi_fa
            g_at = float(norm.pdf(y_pi, mu, sigmaB) /
                         norm.pdf(mu, mu, sigmaB) * DWIDTH)
            ax_b.plot(tick_xs(g_at), [y_pi, y_pi], color=color,
                      lw=PI_TICK_LW, zorder=9, alpha=0.9)

    ax_b.axvline(fa, color="#ddd", lw=0.7, ls="--", zorder=1)
    ax_b.text(fa, YLIM_B[1] + 0.5, f"Age {fa}",
              fontsize=7.5, ha="center", va="bottom",
              color="#444", fontweight="bold")

ax_b.set_xlim(XLIM_B); ax_b.set_ylim(YLIM_B)
ax_b.set_xticks([40, 50, 60, 70]); ax_b.set_yticks([40, 50, 60, 70])
ax_b.set_xlabel("Chronological age (years)", fontsize=9)
ax_b.set_ylabel("Predicted age (years)", fontsize=9)
ax_b.tick_params(labelsize=8.5)

legend_B = [
    Line2D([0], [0], color=CF, lw=1.3, ls="--", alpha=0.8, label="Female"),
    Line2D([0], [0], color=CM, lw=1.3, ls="--", alpha=0.8, label="Male"),
]
ax_b.legend(handles=legend_B, fontsize=8, frameon=False,
            loc="upper left", handlelength=1.8, labelspacing=0.4)


# ---- Panel C: heart failure KM by PI groups ----
GROUPS_KM = [
    ("old",    "Aged heart",     C_OLD,   1.8),
    ("normal", "Normal heart",   C_NORM,  1.4),
    ("young",  "Youthful heart", C_YOUNG, 1.8),
]

for grp, grp_label, color, lw in GROUPS_KM:
    sub      = merged_C[merged_C["pi_group"] == grp]
    followup = np.clip(
        (sub["heart_failure_time"] - sub["age_at_baseline"]).values, 0, T_MAX)
    event    = sub["heart_failure_event"].values.astype(float)
    event[followup >= T_MAX] = 0
    t_grid, cum_inc, ci_lo, ci_hi = km_curve(followup, event, t_max=T_MAX)
    ax_c.plot(t_grid, cum_inc * 100, color=color, lw=lw,
              zorder=4, label=grp_label)
    ax_c.fill_between(t_grid, ci_lo * 100, ci_hi * 100,
                      color=color, alpha=0.12, zorder=3)

ax_c.set_xlim([0, T_MAX]); ax_c.set_ylim(bottom=0)
ax_c.set_xticks([0, 5, 10, 15])
ax_c.set_xlabel("Follow-up (years)", fontsize=9)
ax_c.set_ylabel("Cumulative incidence (%)", fontsize=9)
ax_c.set_title("Heart failure", fontsize=9, fontweight="bold", pad=5)
ax_c.tick_params(labelsize=8.5)
ax_c.legend(frameon=False, loc="upper left", fontsize=8,
            handlelength=1.5, labelspacing=0.4)


# ---- Panel labels ----
for label, ax_l in [("A", ax_a), ("B", ax_b), ("C", ax_c)]:
    ax_l.text(-0.10, 1.06, label, transform=ax_l.transAxes,
              fontsize=12, fontweight="bold", va="top", ha="left")

fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUTPUT_PNG}, {OUTPUT_PDF}")
