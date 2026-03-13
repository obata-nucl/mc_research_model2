from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from pathlib import Path
from src.data import load_raw_expt_spectra
from src.loader import load_eval_results
from src.utils import load_config

CONFIG = load_config()

plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 12

# eval_results = ["N", "E2+_1", "E4+_1", "E6+_1", "E0+_2", "R_4/2", "eps", "kappa", "chi_n"]
# expt_spectra = {(p, n), np.ndarray}

def _spectra_panel_labels(c_beta: float | None) -> tuple[str, str]:
    if c_beta is None:
        left = "(a) IBM-2"
    else:
        left = rf"(a) IBM-2 ($C_{{\beta}}={float(c_beta):.1f}$)"
    right = "(b) Expt."
    return left, right


def _z_panel_label(z: int) -> str | None:
    mapping = {
        60: "(a) Nd",
        62: "(b) Sm",
        64: "(c) Gd",
    }
    return mapping.get(int(z))


def _plot_spectra(pred_data: np.ndarray, expt_data: dict[tuple[int, int], np.ndarray], c_beta: float | None = None, level_labels: list[str] = ["2+_1", "4+_1", "6+_1", "0+_2"], markers: list[str] = ['o', 's', '^', 'D']) -> tuple[plt.Figure, float]:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)
    expt_keys = list(expt_data.keys())
    expt_energies = np.array([expt_data[key] for key in expt_keys])

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    level_labels_tex = {
        "2+_1": r"$2^+_1$",
        "4+_1": r"$4^+_1$",
        "6+_1": r"$6^+_1$",
        "0+_2": r"$0^+_2$",
    }

    left_label, right_label = _spectra_panel_labels(c_beta)
    ax[0].text(
        0.03,
        0.96,
        left_label,
        transform=ax[0].transAxes,
        fontsize=16,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )
    ax[1].text(
        0.03,
        0.96,
        right_label,
        transform=ax[1].transAxes,
        fontsize=16,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )

    sort_idx = np.argsort(pred_data[:, 0].astype(int))
    sorted_pred = pred_data[sort_idx]
    neutron_numbers = sorted_pred[:, 0].astype(int)

    expt_neutrons = np.array([key[1] for key in expt_keys], dtype=int)
    expt_sort_idx = np.argsort(expt_neutrons)
    expt_neutrons = expt_neutrons[expt_sort_idx]
    expt_energies = expt_energies[expt_sort_idx]

    for i in range(len(level_labels)):
        color = colors[i % len(colors)]
        label = level_labels_tex.get(level_labels[i], level_labels[i])

        pred_series = sorted_pred[:, i + 2]
        pred_valid = ~np.isnan(pred_series)
        if np.any(pred_valid):
            ax[0].plot(
                neutron_numbers[pred_valid],
                pred_series[pred_valid],
                marker=markers[i],
                color=color,
                label=label,
                linewidth=2.8,
                markersize=7,
            )

        expt_series = expt_energies[:, i]
        expt_valid = ~np.isnan(expt_series)
        if np.any(expt_valid):
            ax[1].plot(
                expt_neutrons[expt_valid],
                expt_series[expt_valid],
                marker=markers[i],
                color=color,
                label=label,
                linewidth=2.8,
                markersize=7,
            )

    max_energy = 0.0
    for i in range(len(level_labels)):
        pred_series = sorted_pred[:, i + 2]
        expt_series = expt_energies[:, i]
        pred_max = np.nanmax(pred_series) if np.any(~np.isnan(pred_series)) else 0.0
        expt_max = np.nanmax(expt_series) if np.any(~np.isnan(expt_series)) else 0.0
        max_energy = max(max_energy, pred_max, expt_max)
    energy_limit = max(2.0, max_energy * 1.1)

    for a in ax:
        a.set_xlabel("Neutron Number", fontsize=16)
        a.set_ylabel("Energy [MeV]", fontsize=16)
        a.legend(loc="best", fontsize=12)
        a.grid(True, linestyle='--', alpha=0.5)
        a.tick_params(axis="both", which="major", labelsize=16)
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.set_ylim(bottom=0.0)

    # Match Model1 behavior: keep the expt panel with fixed scale in the base plot.
    ax[1].set_ylim(0.0, 2.0)

    fig.tight_layout()
    return fig, energy_limit

def _plot_ratio(pred_data: np.ndarray, expt_data: dict[tuple[int, int], np.ndarray], panel_label: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    sort_idx = np.argsort(pred_data[:, 0].astype(int))
    sorted_pred = pred_data[sort_idx]
    pred_n = sorted_pred[:, 0].astype(int)
    pred_ratio = sorted_pred[:, 6]

    expt_keys = list(expt_data.keys())
    expt_n = np.array([key[1] for key in expt_keys], dtype=int)
    expt_ratio = np.array([expt_data[key][4] for key in expt_keys])
    expt_sort_idx = np.argsort(expt_n)
    expt_n = expt_n[expt_sort_idx]
    expt_ratio = expt_ratio[expt_sort_idx]

    pred_valid = ~np.isnan(pred_ratio)
    if np.any(pred_valid):
        ax.plot(
            pred_n[pred_valid],
            pred_ratio[pred_valid],
            marker='D',
            color="#2A23F3",
            linewidth=2.8,
            markersize=8,
            label="IBM-2",
        )

    expt_valid = ~np.isnan(expt_ratio)
    if np.any(expt_valid):
        ax.plot(
            expt_n[expt_valid],
            expt_ratio[expt_valid],
            marker='D',
            color="#5C006E",
            linestyle="--",
            linewidth=2.6,
            markersize=8,
            label="Expt.",
        )

    ax.set_ylim(1.5, 3.5)
    ax.set_xlabel("Neutron Number", fontsize=14)
    ax.set_ylabel(r"$R_{4/2}$", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best", fontsize=12)

    if panel_label is not None:
        ax.text(
            0.96,
            0.07,
            panel_label,
            transform=ax.transAxes,
            fontsize=24,
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
        )

    fig.tight_layout()
    return fig

def _plot_params(pred_data: np.ndarray, element_name: str = "", labels: dict[str, str] = {"eps": r"$\epsilon$ (MeV)", "kappa": r"$\kappa$ (MeV)", "chi_pi": r"$\chi_{\pi}$", "chi_n": r"$\chi_{\nu}$"}, lims: dict[str, tuple[float, float]] = None, combined: bool = False) -> plt.Figure:
    keys = list(labels.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    unique_Zs = np.sort(np.unique(pred_data[:, 1].astype(int)))
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    # Match Model1 default limits when caller does not pass custom limits.
    if lims is None:
        lims = {
            "eps": (0.0, 3.5),
            "kappa": (-1.0, 0.0),
            "chi_pi": (-1.5, 0.0),
            "chi_n": (-1.5, 0.0),
        }

    color_map = {60: "blue", 62: "red", 64: "green"}
    line_map = {60: "-", 62: "--", 64: ":"}
    marker_map = {60: "o", 62: "s", 64: "^"}

    for i, param_name in enumerate(keys):
        ax = axes_flat[i]
        ylabel = labels[param_name]

        for j, z in enumerate(unique_Zs):
            mask = pred_data[:, 1].astype(int) == z
            if not np.any(mask):
                continue
            z_data = pred_data[mask]
            sort_idx = np.argsort(z_data[:, 0].astype(int))
            sorted_z_data = z_data[sort_idx]

            n_nums = sorted_z_data[:, 0].astype(int)
            vals = sorted_z_data[:, i + 7]
            valid = ~np.isnan(vals)
            if not np.any(valid):
                continue

            symbol = CONFIG["elements"].get(int(z), f"Z={z}")
            if combined:
                c = color_map.get(int(z), ["blue", "red", "green"][j % 3])
                ls = line_map.get(int(z), ["-", "--", ":"][j % 3])
                mk = marker_map.get(int(z), ["o", "s", "^"][j % 3])
                ax.plot(
                    n_nums[valid],
                    vals[valid],
                    marker=mk,
                    linestyle=ls,
                    color=c,
                    label=symbol,
                    alpha=0.7,
                    linewidth=2.6,
                    markersize=7,
                )
            else:
                # Single-Z mode keeps a simple black line as before, but with Model1-like thickness.
                ax.plot(
                    n_nums[valid],
                    vals[valid],
                    marker="o",
                    linestyle="-",
                    color="black",
                    linewidth=2.6,
                    markersize=7,
                )

        ax.text(
            0.03,
            0.95,
            panel_labels[i],
            transform=ax.transAxes,
            fontsize=24,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
        )

        ax.set_xlabel("Neutron Number N" if i >= 2 else "", fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        if param_name in lims:
            ax.set_ylim(lims[param_name])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", labelsize=18, width=1.4)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if combined:
            ax.legend(fontsize=20)

    for j in range(len(keys), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    return fig

def save_fig(fig: plt.Figure, filename: str, save_dir: Path = None, close_fig: bool = True) -> None:
    if save_dir is None:
        save_dir = Path(CONFIG["paths"]["results_dir"])
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_stem = save_dir / filename
    fig.savefig(f"{save_stem}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{save_stem}.pdf", bbox_inches='tight')
    if close_fig:
        plt.close(fig)
    return

def main():
    expt_data = load_raw_expt_spectra(
        CONFIG["nuclei"]["p_min"],
        CONFIG["nuclei"]["p_max"],
        CONFIG["nuclei"]["n_min"],
        CONFIG["nuclei"]["n_max"],
        CONFIG["nuclei"]["p_step"]
    )
    for pattern_name, pred_data in load_eval_results().items():
        # pred_data: [N, Z, E2, E4, E6, E0, R, eps, kappa, chi_n]
        # Split by Z
        unique_Zs = np.unique(pred_data[:, 1].astype(int))
        for z in unique_Zs:
            mask = (pred_data[:, 1].astype(int) == z)
            z_pred_data = pred_data[mask]
            
            # Filter expt data for this Z
            z_expt_data = {k: v for k, v in expt_data.items() if k[0] == z}
            
            if not z_expt_data:
                continue

            element_name = CONFIG["elements"].get(int(z), f"Z={z}")
            save_dir = CONFIG["paths"]["results_dir"] / "images" / pattern_name / str(z)
            c_beta = CONFIG.get("nuclei", {}).get("fixed_C_beta")

            fig_spectra, spectra_limit = _plot_spectra(z_pred_data, z_expt_data, c_beta=c_beta)
            save_fig(fig_spectra, "spectra", save_dir, close_fig=False)
            for axis in fig_spectra.axes:
                axis.set_ylim(0.0, spectra_limit)
            save_fig(fig_spectra, "spectra_common_scale", save_dir)

            g_level_labels = ["2+_1", "4+_1", "6+_1"]
            g_markers = ['o', 's', '^']
            fig_spectra_g, _ = _plot_spectra(
                z_pred_data,
                z_expt_data,
                c_beta=c_beta,
                level_labels=g_level_labels,
                markers=g_markers
            )
            save_fig(fig_spectra_g, "spectra_g", save_dir, close_fig=False)
            for axis in fig_spectra_g.axes:
                axis.set_ylim(0.0, spectra_limit)
            save_fig(fig_spectra_g, "spectra_g_common_scale", save_dir)

            fig_ratio = _plot_ratio(z_pred_data, z_expt_data, panel_label=_z_panel_label(int(z)))
            save_fig(fig_ratio, "ratio", save_dir)

            param_lims = {
                "eps": (0.0, 3.5),
                "kappa": (-1.0, 0.0),
                "chi_pi": (-2.0, 0),
                "chi_n": (-2.0, 0)
            }
            fig_params = _plot_params(z_pred_data, element_name=element_name, lims=param_lims)
            save_fig(fig_params, "params", save_dir)

        # Plot parameters for all elements together (Nd, Sm, Gd)
        target_Zs = [60, 62, 64]
        mask = np.isin(pred_data[:, 1].astype(int), target_Zs)
        if np.any(mask):
            filtered_pred_data = pred_data[mask]
            param_lims_combined = {
                "eps": (0.0, 3.5),
                "kappa": (-1.0, 0.0),
                "chi_pi": (-2.0, 0.0),
                "chi_n": (-2.0, 0.0)
            }
            fig_params_all = _plot_params(filtered_pred_data, lims=param_lims_combined, combined=True)
            save_dir_combined = CONFIG["paths"]["results_dir"] / "images" / pattern_name
            save_fig(fig_params_all, "params_all", save_dir_combined)
    return

if __name__ == "__main__":
    main()
