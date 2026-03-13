from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from pathlib import Path
from src.utils import load_config
from src.plotting.plot import save_fig

CONFIG = load_config()

def _load_loss_csv(csv_path: Path) -> dict[str, np.ndarray]:
    """Load loss.csv and return required columns by name."""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
    except OSError as exc:
        raise RuntimeError(f"Failed to read {csv_path}") from exc

    required_cols = ["epoch", "train_RMSE", "val_RMSE", "lr"]
    missing = [c for c in required_cols if c not in header]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    col_idx = {name: i for i, name in enumerate(header)}

    try:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    except OSError as exc:
        raise RuntimeError(f"Failed to read {csv_path}") from exc

    if data.size == 0:
        raise ValueError(f"No data rows found in {csv_path}")

    if data.ndim == 1:
        data = data[np.newaxis, :]

    return {
        "epoch": data[:, col_idx["epoch"]],
        "train_rmse": data[:, col_idx["train_RMSE"]],
        "val_rmse": data[:, col_idx["val_RMSE"]],
        "lr": data[:, col_idx["lr"]],
    }

def _plot_learning_curve(curves: dict[str, np.ndarray]) -> plt.Figure:
    """Plot train/val RMSE (left axis) and learning rate (right axis)."""
    epochs = curves["epoch"]
    train_rmse = curves["train_rmse"]
    val_rmse = curves["val_rmse"]
    lrs = curves["lr"]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_title("Learning Curve", fontsize=18)
    ax1.set_xlabel("Epoch", fontsize=16)
    ax1.set_ylabel("RMSE", fontsize=16)
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.tick_params(labelsize=16)

    line_train, = ax1.plot(epochs, train_rmse, label="train_RMSE", color="#1f77b4", linewidth=2.8)
    line_val, = ax1.plot(epochs, val_rmse, label="val_RMSE", color="#ff7f0e", linewidth=2.8)

    ax1.text(
        0.8,
        0.75,
        "(b) Model 2",
        transform=ax1.transAxes,
        fontsize=16,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )

    ax2 = ax1.twinx()
    lr_arr = np.asarray(lrs, dtype=float)
    valid = np.isfinite(lr_arr)
    if valid.any():
        # Scale LR to 1e-3 units for readability.
        lr_scaled = lr_arr * 1e3
        line_lr, = ax2.plot(epochs, lr_scaled, color="#2ca02c", alpha=0.45, linewidth=2.8, label="lr")
        y_min = float(np.nanmin(lr_scaled[valid]))
        y_max = float(np.nanmax(lr_scaled[valid]))
        pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05
        ax2.set_ylim(y_min - pad, y_max + pad)
    else:
        line_lr = None

    ax2.set_ylabel(r"Learning Rate ($\times 10^{-3}$)", fontsize=16)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.tick_params(labelsize=16)

    legend_handles = [line_train, line_val]
    if line_lr is not None:
        legend_handles.append(line_lr)
    ax1.legend(handles=legend_handles, loc="best", fontsize=12)

    fig.tight_layout()
    return fig



def main():
    results_training_dir = CONFIG["paths"]["results_dir"] / "training"

    pattern_dirs = sorted(d for d in results_training_dir.iterdir() if d.is_dir())
    if not pattern_dirs:
        print(f"No pattern directories under {results_training_dir}")
        return

    for pattern_dir in pattern_dirs:
        csv_path = pattern_dir / "loss.csv"
        if not csv_path.exists():
            print(f"Skipping {pattern_dir.name}: loss.csv not found")
            continue

        try:
            data = _load_loss_csv(csv_path)
        except Exception as e:
            print(f"Skipping {pattern_dir.name}: failed to load loss history: {e}")
            continue

        fig = _plot_learning_curve(data)

        save_fig(fig, "learning_curve", pattern_dir)
    return

if __name__ == "__main__":
    main()
