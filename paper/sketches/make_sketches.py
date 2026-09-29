"""Schematic sketches of the figures proposed for the Experiments section.

The curves are hand-drawn shapes (logistic and power functions), NOT results:
they show what each figure would plot and which comparison it would make.
"""

import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def logistic(x, mid, scale, top=1.0, floor=0.0):
    return floor + (top - floor) / (1.0 + np.exp((x - mid) / scale))


def label(ax, x, y, text, color, dx=4, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                color=INK, fontsize=8, va="center")
    ax.plot([x], [y], "o", color=color, ms=4)


fig, axes = plt.subplots(4, 2, figsize=(10, 13))
fig.suptitle("SCHEMATIC SKETCHES: expected shapes, not simulation results",
             color="#e34948", fontsize=12, weight="bold")

# 1. Calibration under H0 across architectures
ax = axes[0, 0]
archs = ["linear", "(16)", "(64)", "(32,32)", "(64,64,64)"]
x = np.arange(len(archs))
ax.axhspan(0.95 - 0.022, 0.95 + 0.022, color=GRID)
ax.axhline(0.95, color=MUTED, lw=1, ls="--")
for values, color, name, dx in [
    ([0.95, 0.96, 0.94, 0.95, 0.95], BLUE, "PIC-ANN", -0.15),
    ([0.95, 0.40, 0.25, 0.20, 0.15], ORANGE, "no normalization, same λ", 0.0),
    ([0.30, 0.25, 0.22, 0.20, 0.20], AQUA, "LassoNet (CV)", 0.15),
]:
    ax.plot(x + dx, values, "o", color=color, ms=6, label=name)
ax.set_xticks(x, archs)
ax.set_ylim(0, 1.05)
ax.set_ylabel(r"$\hat P(\hat S=\emptyset)$ under $H_0$")
ax.set_title("E1. Null calibration vs architecture", loc="left", color=INK)
ax.legend(frameon=False, fontsize=7, loc="lower left")

# 2. Phase transition, linear signal
ax = axes[0, 1]
s = np.linspace(0, 40, 200)
for curve, color, name in [
    (logistic(s, 22, 3), BLUE, "PIC-ANN (32,32)"),
    (logistic(s, 24, 3), ORANGE, "sqrt-lasso + PIC"),
    (logistic(s, 15, 4, top=0.55), AQUA, "LassoNet (CV)"),
    (logistic(s, 12, 4, top=0.45), VIOLET, "STG (CV)"),
]:
    ax.plot(s, curve, color=color, lw=2)
    at = {BLUE: (95, -60, -8), ORANGE: (125, 6, 8), AQUA: (15, 4, 8), VIOLET: (15, 4, -8)}[color]
    label(ax, s[at[0]], curve[at[0]], name, color, at[1], at[2])
ax.set_xlabel("number of relevant variables $s$")
ax.set_ylabel("P(exact support recovery)")
ax.set_title("E2a. Phase transition, linear signal", loc="left", color=INK)

# 3. Phase transition, nonlinear signal
ax = axes[1, 0]
for curve, color, name in [
    (logistic(s, 18, 3), BLUE, "PIC-ANN (32,32)"),
    (logistic(s, 2, 2, top=0.3), ORANGE, "sqrt-lasso + PIC"),
    (logistic(s, 10, 4, top=0.5), AQUA, "LassoNet (CV)"),
    (logistic(s, 9, 4, top=0.4), VIOLET, "STG (CV)"),
]:
    ax.plot(s, curve, color=color, lw=2)
    at = {BLUE: (95, 6, 6), ORANGE: (22, 10, -2), AQUA: (40, 4, 8), VIOLET: (62, 6, 6)}[color]
    label(ax, s[at[0]], curve[at[0]], name, color, at[1], at[2])
ax.set_xlabel("number of relevant variables $s$")
ax.set_ylabel("P(exact support recovery)")
ax.set_title("E2b. Phase transition, nonlinear additive signal", loc="left", color=INK)

# 4. Width x depth heatmap
ax = axes[1, 1]
widths = [8, 16, 32, 64, 128, 256]
depths = [1, 2, 3, 4]
grid = np.array([[0.70, 0.85, 0.92, 0.93, 0.93, 0.92],
                 [0.75, 0.90, 0.95, 0.95, 0.94, 0.94],
                 [0.72, 0.88, 0.94, 0.94, 0.93, 0.92],
                 [0.68, 0.85, 0.92, 0.92, 0.91, 0.90]])
im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(widths)), widths)
ax.set_yticks(range(len(depths)), depths)
ax.set_xlabel("width (neurons per hidden layer)")
ax.set_ylabel("depth (hidden layers)")
for i in range(len(depths)):
    for j in range(len(widths)):
        ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if grid[i, j] > 0.6 else INK)
ax.set_title("E4. P(exact recovery) vs width and depth", loc="left", color=INK)

# 5. Fraction of effective parameters vs p
ax = axes[2, 0]
p = np.logspace(1, 4, 50)
ax.plot(p, np.minimum(1, 8 / p + 0.02), color=BLUE, lw=2)
label(ax, p[25], min(1, 8 / p[25] + 0.02), "PIC-ANN", BLUE)
ax.plot(p, np.minimum(1, 40 / p + 0.05), color=AQUA, lw=2)
label(ax, p[25], min(1, 40 / p[25] + 0.05), "LassoNet (CV)", AQUA)
ax.plot(p, np.ones_like(p), color=MUTED, lw=1.5, ls="--")
label(ax, p[40], 1.0, "dense MLP", MUTED, -20, 8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of variables $p$ (s = 5 relevant)")
ax.set_ylabel("fraction of non-zero weights")
ax.set_title("E5. Sparsity of the fitted network", loc="left", color=INK)

# 6. Warm path ablation on interaction-only variables
ax = axes[2, 1]
M = np.arange(1, 11)
ax.plot(M, 1 - 0.9 * np.exp(-(M - 1) / 2.0), "o-", color=BLUE, lw=2, ms=5)
label(ax, 3, 1 - 0.9 * np.exp(-2 / 2.0), "interaction-only variables", BLUE, 6, -6)
ax.plot(M, 0.99 - 0.02 * np.exp(-(M - 1)), "o-", color=ORANGE, lw=2, ms=5)
label(ax, 5, 0.99, "marginal-effect variables", ORANGE, -30, 10)
ax.plot(M, 0.05 + 0.01 * M, "o-", color=MUTED, lw=1.5, ms=4)
label(ax, 6, 0.11, "false positives (mean)", MUTED, -30, 10)
ax.set_xlabel("number of phases $M$ (1 = start at the calibrated level)")
ax.set_ylabel("detection rate")
ax.set_ylim(0, 1.1)
ax.set_title("E3. Warm path and purely nonlinear effects", loc="left", color=INK)

# 7. TPR / FDR trade-off at the returned support
ax = axes[3, 0]
for tpr, fdr, color, name in [
    (0.97, 0.03, BLUE, "PIC-ANN"), (0.99, 0.35, AQUA, "LassoNet (CV)"),
    (0.95, 0.30, VIOLET, "STG (CV)"), (0.85, 0.10, ORANGE, "DeepPINK (q = 0.1)"),
]:
    ax.errorbar(fdr, tpr, xerr=0.03, yerr=0.02, fmt="o", color=color, ms=7, capsize=2)
    ax.annotate(name, (fdr, tpr), xytext=(6, -10), textcoords="offset points", fontsize=8)
ax.set_xlim(0, 0.5)
ax.set_ylim(0.7, 1.02)
ax.set_xlabel("false discovery proportion")
ax.set_ylabel("true positive rate")
ax.set_title("E2c. Returned support, one setting (mean ± sd)", loc="left", color=INK)

# 8. Computational cost
ax = axes[3, 1]
n = np.logspace(2, 4.5, 30)
ax.plot(n, 0.02 * n**0.9, color=BLUE, lw=2)
label(ax, n[-8], 0.02 * n[-8]**0.9, "PIC-ANN (path + MC calibration)", BLUE)
ax.plot(n, 0.02 * 50 * n**0.9, color=AQUA, lw=2)
label(ax, n[-8], 0.02 * 50 * n[-8]**0.9, "LassoNet (5-fold CV path)", AQUA)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("sample size $n$ (p = 1000)")
ax.set_ylabel("wall-clock time (s)")
ax.set_title("E8. Cost of one fit, tuning included", loc="left", color=INK)

for ax in axes.flat:
    if ax is axes[1, 1]:
        continue
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("paper/sketches/experiments_sketch.png", dpi=140)
