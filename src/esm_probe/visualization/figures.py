"""Publication-oriented figure helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from esm_probe.utils.io import ensure_dir


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _localization_probes(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return DeepLoc CV probe rows, excluding baselines and external tests."""

    if metrics.empty:
        return metrics
    frame = metrics.copy()
    if "run_type" in frame.columns:
        frame = frame[frame["run_type"].eq("probe")]
    if "dataset" in frame.columns:
        frame = frame[frame["dataset"].astype(str).str.lower().eq("deeploc2")]
    return frame


def _model_label(row: pd.Series) -> str:
    layer = row.get("layer")
    pooling = row.get("pooling")
    if pd.notna(layer) and pd.notna(pooling):
        return f"Layer {int(layer)} {pooling}"
    if pd.notna(layer):
        return f"Layer {int(layer)}"
    return str(row.get("run_id", "run"))


def _format_layer(value: object) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    return str(int(number)) if number.is_integer() else str(value)


def plot_layer_pooling_heatmap(metrics: pd.DataFrame, path: str | Path, value: str = "macro_f1") -> None:
    """Plot mean score by layer and pooling."""

    metrics = _localization_probes(metrics)
    required = {"layer", "pooling", value}
    if metrics.empty or not required <= set(metrics.columns):
        return
    pivot = metrics.pivot_table(index="layer", columns="pooling", values=value, aggfunc="mean")
    pivot = pivot.sort_index()
    pivot.index = [_format_layer(value) for value in pivot.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title("DeepLoc CV macro F1 by layer and pooling")
    ax.set_xlabel("Pooling")
    ax.set_ylabel("ESM layer")
    _save(fig, path)


def plot_macro_f1_by_layer(metrics: pd.DataFrame, path: str | Path) -> None:
    """Plot macro F1 by ESM layer."""

    metrics = _localization_probes(metrics)
    if metrics.empty or not {"layer", "macro_f1"} <= set(metrics.columns):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = metrics.sort_values(["layer", "pooling", "fold", "seed"], na_position="last").copy()
    metrics["layer_label"] = metrics["layer"].map(lambda value: f"L{_format_layer(value)}")
    sns.barplot(data=metrics, x="layer_label", y="macro_f1", errorbar="sd", ax=ax)
    ax.set_title("DeepLoc CV macro F1 by ESM layer")
    ax.set_xlabel("ESM layer")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.0, min(1.0, max(0.7, float(metrics["macro_f1"].max()) + 0.08)))
    _save(fig, path)


def plot_hpa_external_comparison(metrics: pd.DataFrame, path: str | Path) -> None:
    """Plot external-test comparison."""

    if metrics.empty or not {"dataset", "macro_f1"} <= set(metrics.columns):
        return
    external = metrics[metrics["dataset"].astype(str).str.contains("hpa", case=False, na=False)]
    if "run_type" in external.columns:
        external = external[external["run_type"].eq("external_eval")]
    if external.empty:
        return
    external = external.copy()
    external["model_label"] = external.apply(_model_label, axis=1)
    external = external.sort_values(["layer", "pooling"], na_position="last")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=external, x="model_label", y="macro_f1", errorbar=None, ax=ax)
    for patch in ax.patches:
        height = patch.get_height()
        if pd.notna(height):
            ax.annotate(
                f"{height:.3f}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("HPA external-test comparison")
    ax.set_xlabel("Selected DeepLoc probe")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.0, min(1.0, max(0.35, float(external["macro_f1"].max()) + 0.08)))
    _save(fig, path)


def plot_per_label_mcc_matrix(mcc_rows: pd.DataFrame, path: str | Path) -> None:
    """Plot per-label MCC by run/layer where available."""

    if mcc_rows.empty or not {"run_label", "label", "mcc"} <= set(mcc_rows.columns):
        return
    mcc_rows = mcc_rows.dropna(subset=["mcc"]).copy()
    pivot = mcc_rows.pivot_table(index="run_label", columns="label", values="mcc", aggfunc="mean")
    pivot = pivot.sort_index()
    fig_width = max(7, min(16, 0.6 * len(pivot.columns)))
    fig_height = max(4, min(14, 0.35 * len(pivot.index)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Per-label MCC")
    ax.set_xlabel("Localization label")
    ax.set_ylabel("Run group")
    _save(fig, path)


def plot_training_curves(curves: pd.DataFrame, path: str | Path) -> None:
    """Plot training and validation losses."""

    if not {"epoch", "train_loss", "val_loss"} <= set(curves.columns):
        return
    curves = curves.copy()
    if "run_id" in curves.columns:
        curves["layer"] = curves["run_id"].str.extract(r"probe-l(\d+)-", expand=False)
        curves["curve_group"] = curves["layer"].map(lambda value: f"Layer {value}" if pd.notna(value) else "Probe")
    else:
        curves["curve_group"] = "Probe"
    long_curves = curves.melt(
        id_vars=["epoch", "curve_group"],
        value_vars=["train_loss", "val_loss"],
        var_name="split",
        value_name="loss",
    )
    long_curves["split"] = long_curves["split"].map({"train_loss": "Train", "val_loss": "Validation"})
    long_curves = long_curves.rename(columns={"curve_group": "Probe layer", "split": "Loss split"})
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(
        data=long_curves,
        x="epoch",
        y="loss",
        hue="Probe layer",
        style="Loss split",
        estimator="mean",
        errorbar=None,
        ax=ax,
        dashes={"Train": "", "Validation": (3, 2)},
    )
    ax.set_title("Training curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(title="")
    _save(fig, path)
