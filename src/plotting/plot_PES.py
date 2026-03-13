from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path

from src.data import get_boson_num, load_raw_HFB_energies
from src.loader import load_eval_results
from src.physics import IBM2_PES
from src.plotting.plot import save_fig
from src.utils import load_config

CONFIG = load_config()

plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 15

HFB_COLOR = "#6A0DAD"

_TRAINING_CFG = CONFIG.get("training", {})
_BETA_MIN = _TRAINING_CFG.get("beta_min")
_BETA_MAX = _TRAINING_CFG.get("beta_max")
_BETA_POINTS = _TRAINING_CFG.get("beta_points", 200)

def _calc_PES(params: np.ndarray, n_pi: int, n_nu: int, beta_f_arr: np.ndarray) -> np.ndarray:
    """ calculate PES for one nucleus with given N """
    params = np.asarray(params)
    beta_f_arr = np.asarray(beta_f_arr)

    # Convert to tensors and shape for single set calculation
    params_tensor = torch.from_numpy(params.astype(np.float32)).unsqueeze(0)  # (1, 4)
    beta_f_tensor = torch.from_numpy(beta_f_arr.astype(np.float32)).unsqueeze(0)  # (1, num_beta)
    # Convert n_pi and n_nu to tensors for consistent arithmetic and broadcasting
    n_pi_tensor = beta_f_tensor.new_full(beta_f_tensor.shape, float(n_pi))
    n_nu_tensor = beta_f_tensor.new_full(beta_f_tensor.shape, float(n_nu))

    with torch.no_grad():
        pes_tensor = IBM2_PES(params_tensor, n_pi_tensor, n_nu_tensor, beta_f_tensor)

    # pes_tensor shape is (1, nbeta) -> return 1D numpy array
    return pes_tensor.squeeze(0).numpy()

def _prepare_pes_entry(z: int, n: int, params: np.ndarray, n_pi: int, n_nu: int, expt_curve: np.ndarray) -> dict | None:
    if expt_curve is None or expt_curve.size == 0:
        return None

    curve = np.asarray(expt_curve, dtype=float)

    beta_min = _BETA_MIN if _BETA_MIN is not None else curve[:, 0].min()
    beta_max = _BETA_MAX if _BETA_MAX is not None else curve[:, 0].max()

    mask = (curve[:, 0] >= beta_min) & (curve[:, 0] <= beta_max)
    curve = curve[mask]
    if curve.size == 0:
        return None

    order = np.argsort(curve[:, 0])
    beta_expt = curve[order, 0]
    energy_expt = curve[order, 1]

    beta_grid = np.linspace(beta_min, beta_max, _BETA_POINTS)

    baseline_target = np.interp(0.0, beta_expt, energy_expt, left=np.nan, right=np.nan)
    if np.isnan(baseline_target):
        baseline_target = energy_expt[np.argmin(np.abs(beta_expt))]
    energy_expt_shifted = energy_expt - baseline_target

    target_interp = np.interp(beta_grid, beta_expt, energy_expt_shifted, left=np.nan, right=np.nan)

    pred = _calc_PES(params, n_pi, n_nu, beta_grid)
    zero_idx = np.argmin(np.abs(beta_grid))
    baseline = pred[zero_idx]
    pred = pred - baseline

    if np.all(np.isnan(target_interp)):
        return None

    element_symbol = CONFIG["elements"].get(int(z), "X")
    return {
        "Z": int(z),
        "N": int(n),
        "beta": beta_grid,
        "target": target_interp,
        "pred": pred,
        "element": element_symbol
    }


def _plot_pes_grid(z: int, pes_entries: list[dict], save_dir: Path) -> None:
    if not pes_entries:
        return

    pes_entries.sort(key=lambda item: item["N"])
    total = len(pes_entries)
    cols = 4
    rows = int(np.ceil(total / cols))
    rows = max(1, rows)

    base_w, base_h = 4.8, 4.4
    fig, axes = plt.subplots(rows, cols, figsize=(base_w * cols, base_h * rows), sharex=True, sharey=True)

    if rows == 1:
        axes_array = np.array([axes])
    else:
        axes_array = np.array(axes)
    axes_array = axes_array.reshape(rows, cols)

    axes_flat = axes_array.ravel()

    for idx, (entry, ax) in enumerate(zip(pes_entries, axes_flat)):
        beta = entry["beta"]
        target = entry["target"]
        pred = entry["pred"]

        valid_target = np.isfinite(target)
        if np.any(valid_target):
            ax.plot(beta[valid_target], target[valid_target], linestyle="--", color=HFB_COLOR, label="HFB", linewidth=2.6)
            min_target_idx = np.nanargmin(target[valid_target])
            beta_target = beta[valid_target][min_target_idx]
            energy_target = target[valid_target][min_target_idx]
            ax.plot(
                beta_target,
                energy_target,
                marker="o",
                markersize=11,
                markerfacecolor="white",
                markeredgecolor=HFB_COLOR,
                markeredgewidth=2.4,
                zorder=5,
            )

        ax.plot(beta, pred, linestyle="-", color="black", label="IBM-2", linewidth=2.6)
        min_pred_idx = np.nanargmin(pred)
        ax.plot(
            beta[min_pred_idx],
            pred[min_pred_idx],
            marker="o",
            markersize=10,
            markerfacecolor="red",
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=6,
        )

        mass_number = entry["Z"] + entry["N"]
        symbol = entry["element"]
        ax.set_title(rf"$^{{{mass_number}}}\mathrm{{{symbol}}}$", fontsize=22)
        ax.tick_params(axis="both", which="major", labelsize=16, width=1.4)
        if idx == 0:
            ax.legend(loc="best", fontsize=13)

    for ax in axes_flat[total:]:
        ax.axis('off')

    for ax in axes_array[-1, :]:
        if not ax.get_visible():
            continue
        ax.set_xlabel(r"$\beta$", fontsize=18)
    for ax in axes_array[:, 0]:
        if not ax.get_visible():
            continue
        ax.set_ylabel("Energy [MeV]", fontsize=18)

    fig.tight_layout()
    save_fig(fig, "PES_all", save_dir)

def main():
    # beta_f_arr is no longer fixed globally, but determined per nucleus based on expt data
    expt_PES = load_raw_HFB_energies(
        CONFIG["nuclei"]["p_min"],
        CONFIG["nuclei"]["p_max"],
        CONFIG["nuclei"]["n_min"],
        CONFIG["nuclei"]["n_max"],
        CONFIG["nuclei"]["p_step"],
        CONFIG["nuclei"]["n_step"]
    )
    for pattern_name, pred_data in load_eval_results().items():
        # pred_data: [N, Z, E2, E4, E6, E0, R, eps, kappa, chi_pi, chi_n]
        unique_Zs = np.unique(pred_data[:, 1].astype(int))
        
        for z in unique_Zs:
            mask = (pred_data[:, 1].astype(int) == z)
            z_pred_data = pred_data[mask]
            
            Neutrons = z_pred_data[:, 0].astype(int)
            n_pi = get_boson_num(z)
            N_nu = [get_boson_num(int(n)) for n in Neutrons]
            
            pes_entries: list[dict] = []
            for i, (n, n_nu) in enumerate(zip(Neutrons, N_nu)):
                expt_curve = expt_PES.get((z, n))
                params = z_pred_data[i, 7:]
                entry = _prepare_pes_entry(z, n, params, n_pi, n_nu, expt_curve)
                if entry is not None:
                    pes_entries.append(entry)

            if not pes_entries:
                continue

            save_path = CONFIG["paths"]["results_dir"] / "images" / pattern_name / str(z)
            _plot_pes_grid(z, pes_entries, save_path)
    return

if __name__ == "__main__":
    main()
