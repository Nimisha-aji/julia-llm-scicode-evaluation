"""
Analysis & Figure Generator — Julia LLM Scientific Computing Evaluation
Run this AFTER experiment_runner.py completes.

Usage:  python analyze_results.py
Output: results/figures/          — all paper-ready figures
        results/summary_stats.csv — mean +- SD table for paper
        results/anova_results.txt — ANOVA + Tukey HSD + eta-squared
        results/tukey_results.csv — pairwise Tukey HSD per metric

Expected CSV columns in results/experiment_results.csv:
  model          — model identifier string (see MODEL_LABELS below)
  task           — task identifier string
  domain         — one of: numerical_linear_algebra, computational_mechanics,
                   computational_finance
  sample_index   — integer 1–5
  correctness    — float 0.0 or 1.0
  accuracy_score — float 0.0–1.0  (1 - relative_L2_error, clipped)
  exec_time_ms   — float, execution time in ms (NaN if sample failed)
  memory_mb      — float, peak memory in MB (NaN if sample failed)
  codebleu       — float 0.0–1.0 (0.0 if sample failed)
  consistency    — float 0.0–1.0 (computed per model-task group)
  pass_at_1      — float, per model-task pass@1 estimate
  pass_at_5      — float, per model-task pass@5 estimate
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg
import os
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_PATH = "results/experiment_results.csv"
FIGURES_DIR  = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── model labels — update keys to match your actual CSV values ────────────────
MODEL_LABELS = {
    "llama_3_3_70b":  "Llama 3.3 70B",
    "llama_4_scout":  "Llama 4 Scout",
    "qwen3_32b":      "Qwen3 32B",
    "deepseek_r1":    "DeepSeek R1",
    "mixtral_8x7b":   "Mixtral 8x7B",
    "llama_3_1_8b":   "Llama 3.1 8B",
}
MODEL_ORDER = list(MODEL_LABELS.keys())

# colours in the same order as MODEL_ORDER
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

# metrics used across figures and tests
METRICS = {
    "correctness":    "Correctness Rate",
    "accuracy_score": "Numerical Accuracy",
    "exec_time_ms":   "Exec Time (ms)",
    "codebleu":       "CodeBLEU Score",
    "consistency":    "Consistency Score",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(RESULTS_PATH)
    df["model_label"] = df["model"].map(MODEL_LABELS)
    # drop rows whose model is not in MODEL_LABELS
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    return df


def model_summary(df):
    """Return mean +- std per model for all metrics."""
    rows = []
    for m in MODEL_ORDER:
        sub = df[df["model"] == m]
        row = {"Model": MODEL_LABELS[m]}
        for col, label in METRICS.items():
            vals = sub[col].dropna()
            row[f"{label} mean"] = round(vals.mean(), 3)
            row[f"{label} std"]  = round(vals.std(),  3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


# ── FIGURE 1 — grouped bar chart (mean + error bars) ──────────────────────────

def fig1_grouped_bar(df):
    plot_metrics = {
        "correctness":    "Correctness Rate",
        "accuracy_score": "Numerical Accuracy",
        "exec_time_ms":   "Exec Time (ms)",
        "codebleu":       "CodeBLEU Score",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for idx, (col, label) in enumerate(plot_metrics.items()):
        ax = axes[idx]
        means, stds = [], []
        for m in MODEL_ORDER:
            vals = df[df["model"] == m][col].dropna()
            means.append(vals.mean())
            stds.append(vals.std())

        x = np.arange(len(MODEL_ORDER))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=COLORS, edgecolor='black', linewidth=0.5,
                      error_kw=dict(elinewidth=0.8, ecolor='#444'))

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(means) * 0.015,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER],
                           rotation=20, ha='right', fontsize=9)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle(
        "Comparison of Code-Generation LLMs on Julia Scientific Computing Tasks\n"
        "(mean +- 1 SD across 14 tasks × 5 samples)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_grouped_bar.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {path}")


# ── FIGURE 2 — radar / spider chart ───────────────────────────────────────────

def fig2_radar(df):
    radar_cols   = ["correctness", "accuracy_score", "consistency", "codebleu"]
    radar_labels = ["Correctness", "Numerical\nAccuracy", "Consistency", "CodeBLEU"]

    summary = (df.groupby("model")[radar_cols]
                 .mean()
                 .reindex(MODEL_ORDER))

    # efficiency: invert normalised exec time
    exec_means = df.groupby("model")["exec_time_ms"].mean().reindex(MODEL_ORDER)
    summary["efficiency"] = 1 - (exec_means / exec_means.max())
    radar_cols.append("efficiency")
    radar_labels.append("Efficiency\n(inv. time)")

    # normalise 0–1
    for c in radar_cols:
        col = summary[c]
        summary[c] = (col - col.min()) / (col.max() - col.min() + 1e-9)

    N      = len(radar_cols)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, m in enumerate(MODEL_ORDER):
        vals = summary.loc[m, radar_cols].tolist() + [summary.loc[m, radar_cols[0]]]
        ax.plot(angles, vals, 'o-', linewidth=2,
                label=MODEL_LABELS[m], color=COLORS[i])
        ax.fill(angles, vals, alpha=0.07, color=COLORS[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Profile (Normalised Scores)",
                 fontsize=12, fontweight='bold', pad=22)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=8)
    ax.grid(True)

    path = os.path.join(FIGURES_DIR, "fig2_radar.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {path}")


# ── FIGURE 3 — domain heatmap (correctness) ───────────────────────────────────

def fig3_domain_heatmap(df):
    pivot = (df.pivot_table(index="domain", columns="model",
                            values="correctness", aggfunc="mean")
               [MODEL_ORDER])
    pivot.columns = [MODEL_LABELS[m] for m in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    im = ax.imshow(pivot.values, cmap="YlGn", aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=18, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=9, color='black' if val > 0.4 else 'white')

    plt.colorbar(im, ax=ax, label="Mean Correctness Rate")
    ax.set_title("Correctness Rate by Task Domain and Model", fontweight='bold')
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig3_domain_heatmap.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {path}")


# ── FIGURE 4 — consistency box plot ───────────────────────────────────────────

def fig4_consistency_box(df):
    data_by_model = [df[df["model"] == m]["consistency"].dropna().values
                     for m in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data_by_model, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=1.5))

    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER],
                       rotation=18, ha='right', fontsize=9)
    ax.set_ylabel("Consistency Score")
    ax.set_title("Output Consistency Distribution by Model\n"
                 "(across 14 tasks × 5 samples)", fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    path = os.path.join(FIGURES_DIR, "fig4_consistency_box.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {path}")


# ── FIGURE 5 — pass@k grouped bar ─────────────────────────────────────────────

def fig5_passk_bar(df):
    # aggregate pass@k per model (mean over tasks)
    passk = (df.groupby("model")[["pass_at_1", "pass_at_5"]]
               .mean()
               .reindex(MODEL_ORDER))

    x     = np.arange(len(MODEL_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    b1 = ax.bar(x - width/2, passk["pass_at_1"], width,
                label="Pass@1", color=COLORS, edgecolor='black', linewidth=0.5, alpha=0.9)
    b2 = ax.bar(x + width/2, passk["pass_at_5"], width,
                label="Pass@5", color=COLORS, edgecolor='black', linewidth=0.5, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER],
                       rotation=18, ha='right', fontsize=9)
    ax.set_ylabel("Pass@k Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pass@1 and Pass@5 by Model (mean over 14 tasks)",
                 fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig5_passk_bar.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {path}")


# ── ANOVA + TUKEY HSD + ETA-SQUARED ───────────────────────────────────────────

def eta_squared(groups):
    """Compute eta-squared from list of group arrays."""
    all_vals  = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total   = sum((v - grand_mean)**2 for v in all_vals)
    return ss_between / (ss_total + 1e-12)


def run_anova_tukey(df):
    lines = []
    lines.append("=" * 65)
    lines.append("  ONE-WAY ANOVA + TUKEY HSD — Julia LLM Evaluation")
    lines.append("=" * 65)

    tukey_frames = []

    for col, label in METRICS.items():
        groups = [df[df["model"] == m][col].dropna().values
                  for m in MODEL_ORDER]

        # skip if any group is empty or constant
        if any(len(g) == 0 for g in groups):
            lines.append(f"\n{label}: SKIPPED (empty group)")
            continue

        f_stat, p_value = stats.f_oneway(*groups)
        eta2            = eta_squared(groups)
        sig = ("***" if p_value < 0.001 else
               "**"  if p_value < 0.01  else
               "*"   if p_value < 0.05  else "ns")

        lines.append(f"\n{label}:")
        lines.append(f"  F-statistic = {f_stat:.4f}")
        lines.append(f"  p-value     = {p_value:.6f}  {sig}")
        lines.append(f"  eta-squared = {eta2:.4f}  "
                     f"({'large' if eta2 >= 0.14 else 'medium' if eta2 >= 0.06 else 'small'} effect)")

        # Tukey HSD via pingouin (requires long-format data)
        if p_value < 0.05:
            sub = df[["model", col]].dropna()
            sub = sub[sub["model"].isin(MODEL_ORDER)].copy()
            sub.columns = ["group", "value"]
            try:
                tukey = pg.pairwise_tukey(data=sub, dv="value", between="group")

                # find the p-value column - name varies by pingouin version
                pcol = None
                for candidate in ["p-tukey", "p_tukey", "p-corr", "p_corr", "p-unc", "p_unc", "pval"]:
                    if candidate in tukey.columns:
                        pcol = candidate
                        break

                if pcol is None:
                    lines.append(f"  Tukey HSD ran but p-value column not found. Columns: {list(tukey.columns)}")
                else:
                    tukey.insert(0, "metric", label)
                    tukey["sig"] = tukey[pcol].apply(
                        lambda p: "***" if p < 0.001 else
                                  "**"  if p < 0.01  else
                                  "*"   if p < 0.05  else "ns")
                    tukey_frames.append(tukey)

                    sig_pairs = tukey[tukey[pcol] < 0.05]
                    if len(sig_pairs):
                        lines.append(f"  Significant pairwise differences (Tukey HSD):")
                        for _, row in sig_pairs.iterrows():
                            a = MODEL_LABELS.get(row["A"], row["A"])
                            b = MODEL_LABELS.get(row["B"], row["B"])
                            diffcol = "diff" if "diff" in row else "mean(A)-mean(B)"
                            diffval = row[diffcol] if diffcol in row else float("nan")
                            lines.append(
                                f"    {a} vs {b}: "
                                f"diff={diffval:.4f}, "
                                f"p={row[pcol]:.4f} {row['sig']}"
                            )
                    else:
                        lines.append("  No significant pairwise differences after Tukey HSD.")
            except Exception as e:
                lines.append(f"  Tukey HSD failed: {e}")

    lines.append("\n*** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    lines.append("Effect size (eta2): small >= 0.01, medium >= 0.06, large >= 0.14")

    output = "\n".join(lines)
    print(output)

    anova_path = os.path.join("results", "anova_results.txt")
    with open(anova_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\n✓ ANOVA results saved: {anova_path}")

    if tukey_frames:
        tukey_all = pd.concat(tukey_frames, ignore_index=True)
        tukey_path = os.path.join("results", "tukey_results.csv")
        tukey_all.to_csv(tukey_path, index=False)
        print(f"✓ Tukey HSD table saved: {tukey_path}")


# ── SUMMARY TABLE ──────────────────────────────────────────────────────────────

def generate_summary_table(df):
    rows = []
    for m in MODEL_ORDER:
        sub = df[df["model"] == m]
        row = {"Model": MODEL_LABELS[m]}
        for col, label in METRICS.items():
            vals = sub[col].dropna()
            row[f"{label}"] = f"{vals.mean():.3f} +- {vals.std():.3f}"
        # pass@k averages
        row["Pass@1"] = f"{sub['pass_at_1'].mean():.3f}"
        row["Pass@5"] = f"{sub['pass_at_5'].mean():.3f}"
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("Model")

    csv_path = os.path.join("results", "summary_stats.csv")
    summary.to_csv(csv_path)
    print(f"\n✓ Summary table saved: {csv_path}")
    print("\nSUMMARY TABLE:")
    print(summary.to_string())


# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading results...")
    df = load_data()
    print(f"  Loaded {len(df)} rows across "
          f"{df['model'].nunique()} models and "
          f"{df['task'].nunique()} tasks.\n")

    print("Generating figures...")
    fig1_grouped_bar(df)
    fig2_radar(df)
    fig3_domain_heatmap(df)
    fig4_consistency_box(df)
    fig5_passk_bar(df)

    print("\nRunning ANOVA + Tukey HSD + eta-squared...")
    run_anova_tukey(df)

    print("\nGenerating summary table...")
    generate_summary_table(df)

    print("\n✓ All done.")
    print("  Figures  → results/figures/")
    print("  Stats    → results/anova_results.txt")
    print("  Tukey    → results/tukey_results.csv")
    print("  Summary  → results/summary_stats.csv")
