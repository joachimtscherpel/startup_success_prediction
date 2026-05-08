import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import math
import pickle
import os
import re
import glob
import textwrap
from sklearn.metrics import (
    roc_curve,
    auc,
)
import shap

# ── Display constants ─────────────────────────────────────────────────────────
TARGET_ORDER = [
    "target_has_exit",
    "target_has_round",
    "target_employee_delta_upper_q",
    "target_funding_delta_upper_q",
    "target_valuation_delta_upper_q",
]

METRIC_LABELS = {
    "prevalence":                  "Prevalence",
    "rmse":                        "RMSE",
    "r2":                          "R²",
    "auroc":                       "AUROC",
    "auroc_gap":                   "AUROC Gap",
    "aucpr":                       "AUCPR",
    "ap_gap":                      "AP Gap",
    "ap":                          "AP",
    "norm_ap":                     "norm. AP",
    "precision_pos_at_top10":      "Precision@Top10",
    "f1_pos":                      "F1",
    "f1_pos_gap":                  "F1 Gap",
    "precision_pos":               "Precision",
    "recall_pos":                  "Recall",
    "precision_pos_over_baseline": "Precision / Baseline",
}

DEPENDENT_BASE_PARAMS = {
        "embedding_model":    {"emb", "emb_score"},
        "embedding_pca_dim": {"emb", "emb_score"},
        "scoring_model":     {"score", "emb_score"},
    }

TICK_SIZE     = 12
LABEL_SIZE    = 16
TITLE_SIZE    = 16
SUPTITLE_SIZE = 20


# ── Helpers ───────────────────────────────────────────────────────────────────
def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _format_group_label(group_by: str) -> str:
    return group_by.replace("_", " ").title()


def _wrap_target_label(label, width: int = 16) -> str:
    return textwrap.fill(str(label), width=width)

def wrap_labels(ax, width: int = 10):
    """Wrap long x-axis tick labels to avoid overlap."""
    ticks  = ax.get_xticks()
    labels = [textwrap.fill(lbl.get_text(), width) for lbl in ax.get_xticklabels()]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def _mixed_sort_key(s):
    try:
        return (0, float(s))
    except ValueError:
        return (1, [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(s))])


# ── 1. Load results ───────────────────────────────────────────────────────────
def load_all_results(results_dir: str = "results") -> list:
    """
    Load all per-target fold result files.

    New naming convention:
        fold{N}_round{R}_win{W}_quant{Q}_{target}_results.pkl

    Returns a flat list of result dicts, one per target × fold combination.
    """
    # Broad glob — the target slug may contain underscores so we can't be
    # more specific without knowing the target names up front.
    pattern = os.path.join(results_dir, "fold*_round*_win*_quant*_*_results.pkl")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"⚠️  No results found in {results_dir} (pattern: {pattern})")
        return []

    all_results = []
    for file in files:
        try:
            with open(file, "rb") as f:
                res = pickle.load(f)
            # Validate minimum expected keys
            if "target" not in res:
                print(f"⚠️  Skipping {os.path.basename(file)}: missing 'target' key")
                continue
            all_results.append(res)
        except Exception as e:
            print(f"⚠️  Failed to load {os.path.basename(file)}: {str(e)[:80]}")

    # Sort: buyin_round → window_years → target_quantile → target → fold
    all_results.sort(key=lambda r: (
        r.get("buyin_round", 0),
        r.get("window_years", 0),
        r.get("target_quantile") or 0,   # None → 0, sorts before 0.75, 0.90, 0.95
        r.get("target", ""),
        r.get("fold", 0),
    ))
    return all_results


# ── 2. Aggregate metrics ──────────────────────────────────────────────────────
def aggregate_metrics(all_results: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate per-target metrics_test DataFrames across all result files.

    Each result file now contains metrics for exactly one target, so we
    simply add the file-level metadata columns and concatenate.

    Returns
    -------
    reg_aggregated : pd.DataFrame   (regression rows)
    clf_aggregated : pd.DataFrame   (classification rows)
    """
    regression_dfs     = []
    classification_dfs = []

    for idx, result in enumerate(all_results):
        try:
            metrics_df = result.get("metrics_test", pd.DataFrame()).copy()
            if metrics_df is None or metrics_df.empty:
                print(f"⚠️  Result {idx} ({result.get('target', '?')}): empty metrics_test, skipping")
                continue

            # Attach file-level metadata so every row carries the full context
            metrics_df["fold"]            = result.get("fold", idx)
            metrics_df["buyin_round"]     = result.get("buyin_round")
            metrics_df["window_years"]    = result.get("window_years")
            metrics_df["target_quantile"] = result.get("target_quantile")
            # 'target' is already present as a column from evaluate_predictions,
            # but add/overwrite from the file-level key to be safe.
            metrics_df["target"]          = result.get("target")
            metrics_df["f1_pos_gap"]      = result.get("f1_pos_gap")
            metrics_df["auroc_gap"]      = result.get("auroc_gap")
            metrics_df["ap_gap"]      = result.get("ap_gap")

            if "task" in metrics_df.columns:
                reg_df = metrics_df[metrics_df["task"] == "regression"].copy()
                clf_df = metrics_df[metrics_df["task"] == "classification"].copy()
            else:
                # Infer task from is_clf flag if task column is absent
                is_clf = result.get("is_clf", False)
                reg_df = pd.DataFrame() if is_clf else metrics_df.copy()
                clf_df = metrics_df.copy() if is_clf else pd.DataFrame()

            if not reg_df.empty:
                regression_dfs.append(reg_df)
            if not clf_df.empty:
                classification_dfs.append(clf_df)

        except Exception as e:
            print(f"⚠️  Error processing result {idx}: {str(e)[:80]}")

    reg_aggregated = (
        pd.concat(regression_dfs, ignore_index=True) if regression_dfs else pd.DataFrame()
    )
    clf_aggregated = (
        pd.concat(classification_dfs, ignore_index=True) if classification_dfs else pd.DataFrame()
    )
    return reg_aggregated, clf_aggregated


# ── 3. Regression plots ───────────────────────────────────────────────────────
def plot_regression_metrics_by(
    reg_aggregated: pd.DataFrame,
    group_by: str,
    col: str = "target",
    save_path: str | None = None,
):
    if reg_aggregated.empty:
        print("⚠️  No regression metrics to plot")
        return
    if group_by not in reg_aggregated.columns:
        print(f"⚠️  Column '{group_by}' not found in regression metrics")
        return

    metrics = [m for m in ("rmse", "r2") if m in reg_aggregated.columns]
    if not metrics:
        print("⚠️  No RMSE or R² columns found")
        return

    n_metrics   = len(metrics)
    group_label = _format_group_label(group_by)

    # ── No faceting by target ─────────────────────────────────────────────────
    if col is None or col not in reg_aggregated.columns:
        fig, axes = plt.subplots(1, n_metrics, figsize=(2.5 * n_metrics, 3))
        if n_metrics == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            reg_aggregated.boxplot(column=metric, by=group_by, ax=ax)
            ax.set_title(_metric_label(metric), fontsize=TITLE_SIZE)
            ax.set_xlabel(group_label, fontsize=LABEL_SIZE)
            ax.set_ylabel("")
            ax.tick_params(labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)
            wrap_labels(ax)
            if metric == "r2":
                ax.axhline(0, color="r", linestyle="--", alpha=0.5)
        plt.suptitle(f"Regression Metrics by {group_label}", fontsize=SUPTITLE_SIZE)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        return

    # ── Faceted by target ─────────────────────────────────────────────────────
    target_vals = sorted(reg_aggregated[col].dropna().unique())
    n_targets   = len(target_vals)

    fig, axes = plt.subplots(
        nrows=n_metrics, ncols=n_targets,
        figsize=(2.5 * n_targets, 2.5 * n_metrics),
        sharey="row", squeeze=False,
    )

    for i, metric in enumerate(metrics):
        for j, target_val in enumerate(target_vals):
            ax     = axes[i, j]
            subset = reg_aggregated[reg_aggregated[col] == target_val]

            if subset.empty or metric not in subset.columns or subset[metric].isna().all():
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=TICK_SIZE)
            else:
                subset.boxplot(column=metric, by=group_by, ax=ax)
                ax.grid(True, alpha=0.3)
                wrap_labels(ax, width=8)
                if metric == "r2":
                    ax.axhline(0, color="r", linestyle="--", alpha=0.5)

            ax.set_title(_wrap_target_label(target_val) if i == 0 else "", fontsize=TITLE_SIZE)
            ax.set_xlabel(group_label if i == n_metrics - 1 else "", fontsize=LABEL_SIZE)
            ax.set_ylabel(_metric_label(metric) if j == 0 else "", fontsize=LABEL_SIZE)
            ax.tick_params(labelsize=TICK_SIZE)

    plt.suptitle(f"Regression Metrics by {group_label}", fontsize=SUPTITLE_SIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ── 4. Classification plots ───────────────────────────────────────────────────
CLF_METRIC_CANDIDATES = [
    "prevalence",
    "precision_pos",
    "recall_pos",
    "f1_pos",
    "f1_pos_gap",
    "auroc",
    "auroc_gap",
    "ap",
    "ap_gap",
    "norm_ap",
    "precision_pos_at_top10",
    "precision_pos_over_baseline",
]

def plot_classification_metrics_by(
    clf_aggregated: pd.DataFrame,
    group_by: str,
    col: str = "target",
    save_path: str | None = None,
    exclude_quantile_independent: bool = False,  # ← exclude None-quantile rows
):
    if clf_aggregated.empty:
        print("⚠️  No classification metrics to plot")
        return
    if group_by not in clf_aggregated.columns:
        print(f"⚠️  Column '{group_by}' not found in classification metrics")
        return

    df = clf_aggregated.copy()

    # ── Handle None quantile ──────────────────────────────────────────────────
    if "target_quantile" in df.columns:
        if exclude_quantile_independent:
            # Drop rows where target_quantile is None (quantile-independent targets)
            df = df[df["target_quantile"].notna()]
        else:
            # Keep them but label as "—" so they sort and display cleanly
            df["target_quantile"] = df["target_quantile"].apply(
                lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else x
            )

    # If group_by is target_quantile, ensure clean ordering: — first, then 0.75, 0.90, 0.95
    if group_by == "target_quantile" and "target_quantile" in df.columns:
        quant_order = ["—"] + sorted(
            [v for v in df["target_quantile"].unique() if v != "—"],
            key=lambda x: float(x)
        )
        df["target_quantile"] = pd.Categorical(
            df["target_quantile"], categories=quant_order, ordered=True
        )

    metrics = [m for m in CLF_METRIC_CANDIDATES if m in df.columns]
    if not metrics:
        print("⚠️  No classification metric columns found")
        return

    n_metrics   = len(metrics)
    group_label = _format_group_label(group_by)

    # ── No faceting ───────────────────────────────────────────────────────────
    if col is None or col not in df.columns:
        fig, axes = plt.subplots(1, n_metrics, figsize=(2.5 * n_metrics, 3))
        if n_metrics == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            df.boxplot(column=metric, by=group_by, ax=ax)
            ax.set_title(_metric_label(metric), fontsize=TITLE_SIZE)
            ax.set_xlabel(group_label, fontsize=LABEL_SIZE)
            ax.set_ylabel("")
            ax.tick_params(labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)
            wrap_labels(ax)
        plt.suptitle(f"Classification Metrics by {group_label}", fontsize=SUPTITLE_SIZE)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        return

    # ── Faceted by target ─────────────────────────────────────────────────────
    target_vals = sorted(
        df[col].dropna().unique(),
        key=lambda t: (1 if str(t).endswith("_upper_q") else 0, str(t))
    )
    n_targets   = len(target_vals)

    fig, axes = plt.subplots(
        nrows=n_metrics, ncols=n_targets,
        figsize=(2.5 * n_targets, 2.5 * n_metrics),
        sharey="row", squeeze=False,
    )

    for i, metric in enumerate(metrics):
        for j, target_val in enumerate(target_vals):
            ax     = axes[i, j]
            subset = df[df[col] == target_val]
            if subset.empty or metric not in subset.columns or subset[metric].isna().all():
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=TICK_SIZE)
            else:
                subset.boxplot(column=metric, by=group_by, ax=ax)
                ax.grid(True, alpha=0.3)
                wrap_labels(ax, width=8)
            ax.set_title(_wrap_target_label(target_val) if i == 0 else "", fontsize=TITLE_SIZE)
            ax.set_xlabel(group_label if i == n_metrics - 1 else "", fontsize=LABEL_SIZE)
            ax.set_ylabel(_metric_label(metric) if j == 0 else "", fontsize=LABEL_SIZE)
            ax.tick_params(labelsize=TICK_SIZE)

    plt.suptitle(f"Classification Metrics by {group_label}", fontsize=SUPTITLE_SIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# ── 4. Regression diagnostics ─────────────────────────────────────────────────
def plot_regression_analytics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    target_names: list | None = None,
    save_path: str | None = None,
):
    if target_names is None:
        target_names = y_true.columns.tolist()

    n_targets = len(target_names)
    fig, axes = plt.subplots(
        n_targets, 3,
        figsize=(15, 5 * n_targets),
        constrained_layout=True,
    )
    if n_targets == 1:
        axes = axes.reshape(1, -1)

    for idx, target in enumerate(target_names):
        if target not in y_true.columns or target not in y_pred.columns:
            print(f"⚠️  Target '{target}' not found in y_true or y_pred, skipping")
            continue

        mask  = y_true[target].notna() & y_pred[target].notna()
        y_t   = y_true.loc[mask, target].values
        y_p   = y_pred.loc[mask, target].values
        resid = y_t - y_p

        # Scatter
        ax = axes[idx, 0]
        ax.scatter(y_t, y_p, alpha=0.5, edgecolors="k", linewidth=0.5)
        lo, hi = min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")
        ax.set_xlabel("True Value", fontsize=LABEL_SIZE)
        ax.set_ylabel("Predicted Value", fontsize=LABEL_SIZE)
        ax.set_title(f"{target}: Predictions vs True", fontsize=TITLE_SIZE)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Residuals vs predicted
        ax = axes[idx, 1]
        ax.scatter(y_p, resid, alpha=0.5, edgecolors="k", linewidth=0.5)
        ax.axhline(0, color="r", linestyle="--", lw=2)
        ax.set_xlabel("Predicted Value", fontsize=LABEL_SIZE)
        ax.set_ylabel("Residual", fontsize=LABEL_SIZE)
        ax.set_title(f"{target}: Residuals", fontsize=TITLE_SIZE)
        ax.grid(True, alpha=0.3)

        # Residual histogram
        ax = axes[idx, 2]
        ax.hist(resid, bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(resid.mean(), color="r", linestyle="--", lw=2,
                   label=f"Mean: {resid.mean():.4f}")
        ax.set_xlabel("Residual", fontsize=LABEL_SIZE)
        ax.set_ylabel("Frequency", fontsize=LABEL_SIZE)
        ax.set_title(f"{target}: Residual Distribution", fontsize=TITLE_SIZE)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Regression Analytics", fontsize=SUPTITLE_SIZE)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved regression analytics to {save_path}")
    plt.show()


# ── 6. ROC curves ─────────────────────────────────────────────────────────────
def plot_roc_curves(
    y_true: pd.DataFrame,
    y_proba: pd.DataFrame | np.ndarray,
    target_names: list | None = None,
    save_path: str | None = None,
):
    if isinstance(y_proba, np.ndarray):
        y_proba = pd.DataFrame(
            y_proba if y_proba.ndim == 2 else y_proba.reshape(-1, 1),
            columns=target_names or [f"Target_{i}" for i in range(y_proba.shape[-1])],
        )

    if target_names is None:
        target_names = y_true.columns.tolist()

    n_targets = len(target_names)
    n_cols    = min(3, n_targets)
    n_rows    = math.ceil(n_targets / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 5 * n_rows),
                              constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    for idx, target in enumerate(target_names):
        ax = axes_flat[idx]
        try:
            y_t = y_true[target].values
            y_p = (
                y_proba[target].values if target in y_proba.columns
                else y_proba.iloc[:, idx].values
            )
            # Drop NaN pairs
            mask = ~(np.isnan(y_t) | np.isnan(y_p))
            y_t, y_p = y_t[mask], y_p[mask]

            fpr, tpr, _ = roc_curve(y_t, y_p)
            auc_score   = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.4f}")
            ax.plot([0, 1], [0, 1], "r--", lw=2, label="Random")
            ax.set_xlabel("False Positive Rate", fontsize=LABEL_SIZE)
            ax.set_ylabel("True Positive Rate", fontsize=LABEL_SIZE)
            ax.set_title(f"{target}: ROC Curve", fontsize=TITLE_SIZE)
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:60]}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=TICK_SIZE)
            ax.set_title(f"{target}: Error", fontsize=TITLE_SIZE)

    for idx in range(n_targets, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.suptitle("ROC Curves – Classification Targets", fontsize=SUPTITLE_SIZE)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved ROC curves to {save_path}")
    plt.show()


# ── 7. Hyperparameter distribution ───────────────────────────────────────────
def plot_best_params_distribution(
    results_dir: str,
    param_grid: dict | None = None,
    ncols: int = 3,
    figsize_per_subplot: tuple = (5, 4),
    save_path: str | None = None,
    targets: list[str] | str | None = None,   # ← new
):
    """
    Plot the distribution of best hyperparameters selected across all
    per-target fold result files.

    Parameters
    ----------
    targets : str | list[str] | None
        If None, aggregate across all targets (original behaviour).
        If str or list of str, filter to those targets only.

    New file naming:
        fold{N}_round{R}_win{W}_quant{Q}_{target}_results.pkl
    """
    # Normalise targets to a set for fast lookup
    if isinstance(targets, str):
        targets = [targets]
    target_filter = set(targets) if targets is not None else None

    pattern      = os.path.join(results_dir, "fold*_round*_win*_quant*_*_results.pkl")
    result_files = sorted(glob.glob(pattern))

    if not result_files:
        print(f"⚠️  No result files found matching: {pattern}")
        return

    all_best_params = []
    for file in result_files:
        # Extract target name from filename before loading
        # pattern: fold{N}_round{R}_win{W}_quant{Q}_{target}_results.pkl
        basename        = os.path.basename(file)
        basename_no_ext = os.path.splitext(basename)[0]
        parts           = basename_no_ext.split("_")
        try:
            results_idx = parts.index("results")
            # everything between quant{Q} and "results" is the target name
            quant_idx   = next(i for i, p in enumerate(parts) if p.startswith("quant"))
            target_name = "_".join(parts[quant_idx + 1 : results_idx])
        except (ValueError, StopIteration):
            target_name = None

        if target_filter is not None and target_name not in target_filter:
            continue

        try:
            with open(file, "rb") as f:
                data = pickle.load(f)
            if "best_params" in data and data["best_params"]:
                entry = dict(data["best_params"])
                entry["__target__"] = target_name   # carry target for title
                all_best_params.append(entry)
            else:
                print(f"⚠️  'best_params' missing or empty in {basename}")
        except Exception as e:
            print(f"⚠️  Could not load {basename}: {str(e)[:60]}")

    if not all_best_params:
        print("⚠️  No best_params found for the requested targets.")
        return

    df_params  = pd.DataFrame(all_best_params)
    all_params = sorted(param_grid.keys() if param_grid else
                        [c for c in df_params.columns if c != "__target__"])

    # Build plot title
    if target_filter is None:
        suptitle = "Best Hyperparameter Distributions — All Targets"
    elif len(target_filter) == 1:
        suptitle = f"Best Hyperparameter Distributions — {next(iter(target_filter))}"
    else:
        suptitle = "Best Hyperparameter Distributions — " + ", ".join(sorted(target_filter)) 

    if param_grid is not None:
        input_type_keys = [k for k in param_grid if k.endswith("input_type")]
        input_type_col  = input_type_keys[0] if input_type_keys else None
        if not input_type_keys:
            print("⚠️  No 'input_type' key found in param_grid; conditional filtering disabled.")

        dependent_full_map = {}
        for base, required_vals in DEPENDENT_BASE_PARAMS.items():
            matches = [k for k in param_grid if k.endswith(base)]
            if matches:
                dependent_full_map[matches[0]] = required_vals
    else:
        input_type_col     = "input_type" if "input_type" in df_params.columns else None
        dependent_full_map = {b: v for b, v in DEPENDENT_BASE_PARAMS.items()
                               if b in df_params.columns}

    has_input_type = input_type_col is not None and input_type_col in df_params.columns

    params_to_plot = []
    for param in all_params:
        if has_input_type and param in dependent_full_map:
            required_vals = dependent_full_map[param]          # now a set
            if param in df_params.columns:
                mask   = df_params[input_type_col].isin(required_vals)  # ← isin instead of ==
                series = df_params.loc[mask, param].astype(str)
            else:
                series = pd.Series(dtype=str)
        else:
            series = df_params[param].astype(str) if param in df_params.columns else pd.Series(dtype=str)

        params_to_plot.append((param, series))

    n_params = len(params_to_plot)
    if n_params == 0:
        print("⚠️  No parameters to plot.")
        return

    nrows = math.ceil(n_params / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per_subplot[0], nrows * figsize_per_subplot[1]),
        squeeze=False,
    )

    for idx, (param, series) in enumerate(params_to_plot):
        ax     = axes[idx // ncols][idx % ncols]
        counts = series.value_counts() if not series.empty else pd.Series(dtype=int)

        if param_grid is not None and param in param_grid:
            grid_vals = [str(v) for v in param_grid[param]]
            counts    = counts.reindex(grid_vals, fill_value=0)

        if not counts.empty:
            counts = counts.loc[sorted(counts.index, key=_mixed_sort_key)]
            counts.plot(kind="bar", ax=ax, color="skyblue", edgecolor="white")
        else:
            ax.text(0.5, 0.5, "No occurrences", ha="center", va="center",
                    transform=ax.transAxes, fontsize=TICK_SIZE, style="italic")
            ax.set_xticks([])

        ax.set_title(textwrap.fill(param, width=25), fontsize=TITLE_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel("Count", fontsize=LABEL_SIZE)
        ax.tick_params(axis="x", rotation=45, labelsize=TICK_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_SIZE)
        wrap_labels(ax)

    for idx in range(n_params, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.suptitle(suptitle, fontsize=SUPTITLE_SIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✓ Saved hyperparameter distributions to {save_path}")
    plt.show()
    
def print_metrics(metrics_df: pd.DataFrame, title: str | None = None) -> None:
    """Pretty-print a metrics DataFrame from evaluate_predictions."""
    if metrics_df is None or metrics_df.empty:
        print(f"  ⚠ No metrics to display{f' for {title}' if title else ''}")
        return

    if title:
        print(f"\n  {'─'*60}")
        print(f"  {title}")
        print(f"  {'─'*60}")

    numeric_cols = metrics_df.select_dtypes(include="number").columns
    formatted    = metrics_df.copy()
    formatted[numeric_cols] = formatted[numeric_cols].map(
        lambda x: f"{x:.4f}" if pd.notna(x) else "—"
    )
    print(formatted.to_string(index=False, justify="left"))
    
def plot_gap_by_param(
    results_dir: str,
    param_grid: dict | None = None,
    figsize_per_subplot: tuple = (2.5, 2.5),
    save_path: str | None = None,
    targets: list[str] | str | None = None,
    gap_col: str = "f1_pos_gap",
):
    """
    For each hyperparameter (rows) × target (columns), plot the distribution
    of a gap metric (train−test) across folds, broken down by parameter value.

    Supports any gap column present in the result files, e.g.:
        "f1_pos_gap", "auroc_gap", "ap_gap".

    Parameters
    ----------
    results_dir : str
        Directory containing fold result .pkl files.
    param_grid : dict | None
        TUNABLE_PARAMETERS dict. Used to order x-axis values canonically.
    figsize_per_subplot : tuple
        (width, height) per subplot cell.
    save_path : str | None
        If given, saves the figure here.
    targets : str | list[str] | None
        Filter to specific targets. None = all targets.
    gap_col : str
        Column name for the gap metric.
        Common values: "f1_pos_gap", "auroc_gap", "ap_gap".
    """
    # ── Normalise target filter ───────────────────────────────────────────────
    if isinstance(targets, str):
        targets = [targets]
    target_filter = set(targets) if targets is not None else None

    # ── Load files ────────────────────────────────────────────────────────────
    pattern      = os.path.join(results_dir, "fold*_round*_win*_quant*_*_results.pkl")
    result_files = sorted(glob.glob(pattern))

    if not result_files:
        print(f"⚠️  No result files found matching: {pattern}")
        return

    records = []
    for file in result_files:
        basename        = os.path.basename(file)
        basename_no_ext = os.path.splitext(basename)[0]
        parts           = basename_no_ext.split("_")
        try:
            results_idx = parts.index("results")
            quant_idx   = next(i for i, p in enumerate(parts) if p.startswith("quant"))
            target_name = "_".join(parts[quant_idx + 1 : results_idx])
        except (ValueError, StopIteration):
            target_name = None

        if target_filter is not None and target_name not in target_filter:
            continue

        try:
            with open(file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"⚠️  Could not load {basename}: {str(e)[:60]}")
            continue

        best_params = data.get("best_params")
        if not best_params:
            continue

        # ── Extract gap metric ────────────────────────────────────────────────
        gap_value = None
        for key in [gap_col, "metrics", "test_metrics", "cv_results"]:
            candidate = data.get(key)
            if candidate is None:
                continue
            if key == gap_col:
                gap_value = candidate
                break
            if isinstance(candidate, dict) and gap_col in candidate:
                gap_value = candidate[gap_col]
                break
            if isinstance(candidate, pd.DataFrame) and gap_col in candidate.columns:
                gap_value = candidate[gap_col].mean()
                break

        if gap_value is None:
            for v in data.values():
                if isinstance(v, dict) and gap_col in v:
                    gap_value = v[gap_col]
                    break

        if gap_value is None:
            print(f"⚠️  '{gap_col}' not found in {basename}, skipping.")
            continue

        record = dict(best_params)
        record["__target__"] = target_name
        record[gap_col]      = float(gap_value)
        records.append(record)

    if not records:
        print("⚠️  No records with both best_params and gap metric found.")
        return

    df = pd.DataFrame(records)

    # ── Determine params to plot ──────────────────────────────────────────────
    skip_cols  = {"__target__", gap_col}
    all_params = sorted(
        param_grid.keys() if param_grid else
        [c for c in df.columns if c not in skip_cols]
    )

    if param_grid is not None:
        input_type_keys = [k for k in param_grid if k.endswith("input_type")]
        input_type_col  = input_type_keys[0] if input_type_keys else None
        dependent_full_map = {}
        for base, required_vals in DEPENDENT_BASE_PARAMS.items():
            matches = [k for k in param_grid if k.endswith(base)]
            dependent_full_map[matches[0] if matches else base] = required_vals
    else:
        input_type_col     = next((c for c in df.columns if c.endswith("input_type")), None)
        dependent_full_map = dict(DEPENDENT_BASE_PARAMS)

    has_input_type = input_type_col is not None and input_type_col in df.columns

    # ── Build (param, subset_df) pairs ────────────────────────────────────────
    params_to_plot = []
    for param in all_params:
        if param in skip_cols or param not in df.columns:
            continue
        if has_input_type and param in dependent_full_map:
            required_vals = dependent_full_map[param]
            mask          = df[input_type_col].isin(required_vals)
            subset        = df.loc[mask, [param, gap_col, "__target__"]].copy()
        else:
            subset = df[[param, gap_col, "__target__"]].copy()

        subset[param] = subset[param].astype(str)
        if subset.empty or subset[gap_col].isna().all():
            continue
        params_to_plot.append((param, subset))

    if not params_to_plot:
        print("⚠️  No parameters to plot.")
        return

    # ── Layout: params = rows, targets = columns ──────────────────────────────
    target_vals = sorted(
        df["__target__"].dropna().unique(),
        key=lambda t: (1 if str(t).endswith("_upper_q") else 0, str(t)),
    )
    n_params  = len(params_to_plot)
    n_targets = len(target_vals)

    fig, axes = plt.subplots(
        nrows=n_params,
        ncols=n_targets,
        figsize=(figsize_per_subplot[0] * n_targets, figsize_per_subplot[1] * n_params),
        sharey="row",
        squeeze=False,
    )

    gap_label = _metric_label(gap_col)

    for i, (param, subset) in enumerate(params_to_plot):
        # canonical x-axis order
        if param_grid is not None and param in param_grid:
            ordered_vals = [str(v) for v in param_grid[param]]
        else:
            ordered_vals = sorted(subset[param].unique(), key=_mixed_sort_key)

        for j, target_val in enumerate(target_vals):
            ax    = axes[i, j]
            t_sub = subset[subset["__target__"] == target_val].copy()

            if t_sub.empty or t_sub[gap_col].isna().all():
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=TICK_SIZE)
            else:
                present_vals = [v for v in ordered_vals if v in set(t_sub[param])]

                val_to_code = {v: k + 1 for k, v in enumerate(present_vals)}

                t_sub.boxplot(column=gap_col, by=param, ax=ax)
                ax.set_xlabel("")
                ax.set_xticks(range(1, len(present_vals) + 1))
                ax.set_xticklabels(
                    [textwrap.fill(v, width=5) for v in present_vals],
                    rotation=0, ha="center", fontsize=TICK_SIZE,
                )

                ax.grid(True, alpha=0.3)

            ax.set_title(
                _wrap_target_label(target_val) if i == 0 else "",
                fontsize=TITLE_SIZE,
            )

            ax.set_ylabel(
                textwrap.fill(param, width=14) if j == 0 else "",
                fontsize=LABEL_SIZE,
            )
            ax.tick_params(labelsize=TICK_SIZE)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    if target_filter is None or len(target_filter) > 1:
        suptitle = f"{gap_label} by Hyperparameter — All Targets"
    else:
        suptitle = f"{gap_label} by Hyperparameter — {next(iter(target_filter))}"

    plt.suptitle(suptitle, fontsize=SUPTITLE_SIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✓ Saved to {save_path}")
    plt.show()

GROUP_PATTERNS = {
    "scores":              "score",
    "embeddings":          "embedding",
    "tby":                 "tby",
    "founder":             ("founder", "universities"),
    "past":                "past",
    "categorical_company": ("hq_country", "industries", "client_focus", "revenue_model", "income_streams"),
}

def load_shap_data(results_dir: str) -> tuple[dict, dict, dict]:
    """Load and combine SHAP values across folds for all targets in results_dir."""
    all_data = {}

    for res_file in sorted(glob.glob(os.path.join(results_dir, "data", "fold*_results.pkl"))):
        with open(res_file, "rb") as f:
            res = pickle.load(f)

        target = res.get("target")
        if target is None:
            print(f"⚠ Skipping {res_file}: missing 'target' key")
            continue

        shap_record = res.get("shap_results", {}).get(target)
        if shap_record is None:
            print(f"⚠ No SHAP data for {target} in {os.path.basename(res_file)}, skipping")
            continue

        required_keys = {"shap_values", "feature_names", "feature_matrix"}
        if not required_keys.issubset(shap_record.keys()):
            print(f"⚠ SHAP record for {target} missing keys {required_keys - shap_record.keys()}, skipping")
            continue

        all_data.setdefault(target, []).append(shap_record)

    if not all_data:
        print(f"⚠ No valid SHAP data found in {results_dir}")
        return {}, {}, {}

    print(f"Loaded SHAP data for {len(all_data)} targets from {results_dir}")

    combined_shap, combined_features, feature_names_per_target = {}, {}, {}

    for target, records in all_data.items():
        all_feat_names = sorted({fn for rec in records for fn in rec["feature_names"]})
        feature_names_per_target[target] = all_feat_names

        shap_arrays, feat_arrays = [], []
        for rec in records:
            shap_arrays.append(
                pd.DataFrame(rec["shap_values"], columns=rec["feature_names"])
                  .reindex(columns=all_feat_names, fill_value=0.0).values
            )
            feat_arrays.append(
                pd.DataFrame(rec["feature_matrix"], columns=rec["feature_names"])
                  .reindex(columns=all_feat_names, fill_value=0.0).values
            )

        combined_shap[target]     = np.vstack(shap_arrays)
        combined_features[target] = np.vstack(feat_arrays)

    return combined_shap, combined_features, feature_names_per_target

def compute_shap_display_df(combined_shap: dict, feature_names_per_target: dict) -> pd.DataFrame:
    """Compute the SHAP group importance display DataFrame (values in %)."""
    rows = []
    for target in sorted(combined_shap.keys()):
        shap_vals  = combined_shap[target]
        feat_names = feature_names_per_target[target]
        mean_abs       = np.mean(np.abs(shap_vals), axis=0)
        total_abs_shap = mean_abs.sum()

        row = {"target": target}
        for group_name, pattern in GROUP_PATTERNS.items():
            if isinstance(pattern, tuple):
                mask = np.array([any(p in fn.lower() for p in pattern) for fn in feat_names])
            else:
                mask = np.array([pattern in fn.lower() for fn in feat_names])
            group_abs       = mean_abs[mask].sum()
            row[group_name] = group_abs / total_abs_shap if total_abs_shap > 0 else 0.0
        rows.append(row)

    df = pd.DataFrame(rows).set_index("target")
    df = df.mul(100).round(1).rename(columns={
        "embeddings":          r"Embeddings (\%)",
        "founder":             r"Founders (\%)",
        "categorical_company": r"Cat. Company (\%)",
        "past":                r"Past (\%)",
        "scores":              r"Scores (\%)",
        "tby":                 r"10y TBY (\%)",
    })
    group_cols = [c for c in df.columns]
    group_cols_sorted = sorted(group_cols, key=lambda c: df[c].mean(), reverse=True)
    return df[group_cols_sorted]

def plot_beeswarm(
    shap_vals,
    feat_vals,
    feature_names,
    title,
    save_path,
    max_display=30,
    alpha=0.12,
    dot_size=12,
    fig_width=12,
):
    """
    Wrapper around shap.summary_plot that applies consistent styling,
    saves to SVG, and closes the figure afterwards.

    Parameters
    ----------
    shap_vals     : np.ndarray, shape (n_samples, n_features)
    feat_vals     : np.ndarray, shape (n_samples, n_features) — used for colour scale
    feature_names : list[str]
    title         : str
    save_path     : str
    """
    n_features = min(max_display, shap_vals.shape[1])
    fig_height  = max(6, n_features * 0.6)

    plt.figure(figsize=(fig_width, fig_height))
    shap.summary_plot(
        shap_vals,
        feat_vals,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None,
    )

    ax = plt.gca()

    # Dot styling
    for coll in ax.collections:
        coll.set_alpha(alpha)
        coll.set_sizes([dot_size])
        coll.set_rasterized(True)

    # Slightly widen x-axis so edge dots aren't clipped
    xl  = ax.get_xlim()
    pad = (xl[1] - xl[0]) * 0.20
    ax.set_xlim(xl[0] - pad, xl[1] + pad)

    # Light vertical grid
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)

    plt.title(title, fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

def save_latex_metrics_table(
    avg_df: pd.DataFrame,
    group_label: str,
    save_path: str | None = None,
    fmt: str = ".2f",
):
    def escape_underscore(s: str) -> str:
        return s.replace("_", "\\_")

    col_headers = "\n".join(
        f"& \\rotatebox{{90}}{{\\textbf{{{METRIC_LABELS.get(m, escape_underscore(m))}}}}}" for m in avg_df.columns
    )
    avg_df = avg_df.reindex(
        [t for t in TARGET_ORDER if t in avg_df.index] +
        [t for t in avg_df.index if t not in TARGET_ORDER]
    )
    rows = []
    for target, row in avg_df.iterrows():
        vals = " & ".join(f"{v:{fmt}}" for v in row)
        rows.append(f"  \\texttt{{{escape_underscore(target)}}} & {vals} \\\\")

    n_cols   = 1 + len(avg_df.columns)
    col_spec = "l" + " c" * (n_cols - 1)

    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Average Classification Metrics by {group_label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{@{{}} {col_spec} @{{}}}}",
        r"\toprule",
        r"\textbf{Target}",
        col_headers + " \\\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    table_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path)) if save_path else "result_table.tex"
    with open(table_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"📊 LaTeX table saved → {table_path}")

def save_latex_shap_importance_table(
    display_df: pd.DataFrame,
    results_dir: str,
    filename: str = "shap_feature_group_importance.tex",
    fmt: str = ".1f",
    caption: str = "Average SHAP Feature Importance by Target Group",
    label: str = "tab:shap_feature_group_importance",
):
    def escape_underscore(s: str) -> str:
        return s.replace("_", "\\_")

    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, filename)

    # ------------------------------------------------------------------ #
    # Sort columns by total sum (highest to left)                       #
    # For raw importance (not delta) we use sum, but consider absolute?  #
    # Original used .sum(). We'll keep as is.                           #
    # ------------------------------------------------------------------ #
    display_df = display_df[display_df.sum().sort_values(ascending=False).index]

    # ------------------------------------------------------------------ #
    # Reorder rows to TARGET_ORDER when available                       #
    # ------------------------------------------------------------------ #
    try:
        ordered_index = (
            [t for t in TARGET_ORDER if t in display_df.index]
            + [t for t in display_df.index if t not in TARGET_ORDER]
        )
        display_df = display_df.reindex(ordered_index)
    except NameError:
        pass

    # ------------------------------------------------------------------ #
    # Build column headers with escaping & rotation                     #
    # ------------------------------------------------------------------ #
    col_headers = [escape_underscore(str(col)) for col in display_df.columns]
    header_row = " & ".join([r"\rotatebox{90}{\textbf{%s}}" % h for h in col_headers])
    
    # ------------------------------------------------------------------ #
    # Data rows                                                          #
    # ------------------------------------------------------------------ #
    rows = []
    for target, row in display_df.iterrows():
        target_escaped = escape_underscore(str(target))
        vals = " & ".join(f"{v:{fmt}}" for v in row)
        rows.append(f"  \\texttt{{{target_escaped}}} & {vals} \\\\")
    
    # ------------------------------------------------------------------ #
    # Column specification: first column left-aligned, others centered  #
    # ------------------------------------------------------------------ #
    n_cols = len(display_df.columns)
    col_spec = "l " + "c " * n_cols
    col_spec = col_spec.strip()
    
    # ------------------------------------------------------------------ #
    # Assemble full table                                                #
    # ------------------------------------------------------------------ #
    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{} " + col_spec + r" @{}}",
        r"\toprule",
        r"\textbf{Target} & " + header_row + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(save_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")

    print(f"📊 LaTeX SHAP table saved → {save_path}")


def save_latex_shap_importance_delta_table(
    display_df_a: pd.DataFrame,
    display_df_b: pd.DataFrame,
    results_dir: str,
    filename: str = "shap_feature_group_importance_delta.tex",
    label_a: str = "Regular CV",
    label_b: str = "Walk-Forward CV",
    fmt: str = "+.1f",
):
    """
    Compute (display_df_a − display_df_b) and save a signed LaTeX table.

    Parameters
    ----------
    display_df_a : pd.DataFrame   Minuend   (e.g. CV run).
    display_df_b : pd.DataFrame   Subtrahend (e.g. baseline run).
    results_dir  : str            Output directory.
    filename     : str            Output filename.
    label_a      : str            Human-readable name for A (used in caption).
    label_b      : str            Human-readable name for B (used in caption).
    fmt          : str            Format spec for signed numeric cells (default "+.1f").
    """
    # Intersection of columns
    shared_cols = display_df_a.columns.intersection(display_df_b.columns)
    delta_df = display_df_a[shared_cols] - display_df_b[shared_cols]
    
    # Sort columns by total absolute impact (largest deltas first)
    delta_df = delta_df[delta_df.abs().sum().sort_values(ascending=False).index]
    
    # Use the same table generator with appropriate caption
    save_latex_shap_importance_table(
        display_df=delta_df,
        results_dir=results_dir,
        filename=filename,
        fmt=fmt,
        caption=f"SHAP Feature Group Importance — Delta ({label_a} $-$ {label_b})",
        label="tab:shap_feature_group_importance_delta",
    )