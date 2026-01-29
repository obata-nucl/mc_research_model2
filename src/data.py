from __future__ import annotations
import numpy as np
import torch

from src.utils import load_config, load_scaler

CONFIG = load_config()
# SCALER = load_scaler(CONFIG)  <-- Removed global load to avoid FileNotFoundError during training setup

def load_raw_HFB_energies(p_min: int, p_max: int, n_min: int, n_max: int, p_step: int, n_step: int) -> dict[tuple[int, int], np.ndarray]:
    raw_dir = CONFIG["paths"]["raw_dir"]
    data = {}
    for p in range(p_min, p_max + 1, p_step):
        file_dir = raw_dir / str(p)
        for n in range(n_min, n_max + 1, n_step):
            file_path = file_dir / f"{n}.csv"
            try:
                data[(p, n)] = np.loadtxt(file_path, delimiter=',', skiprows=1)
            except Exception as e:
                print(f"Error loading data for N = {n} from {file_path}: {e}")
                data[(p, n)] = None
    return data

def load_raw_expt_spectra(p_min: int, p_max: int, n_min: int, n_max: int, p_step: int) -> dict[tuple[int, int], np.ndarray]:
    """ load experimental spectra from expt.csv file """
    raw_dir = CONFIG["paths"]["raw_dir"]
    spectra: dict[tuple[int, int], np.ndarray] = {}

    for p in range(p_min, p_max + 1, p_step):
        file_dir = raw_dir / str(p)
        file_path = file_dir / "expt.csv"
        try:
            arr = np.loadtxt(file_path, delimiter=',', skiprows=1)
        except Exception as e:
            print(f"[WARN] Failed to load expt.csv for Z={p} at {file_path}: {e}")
            continue

        n_arr = arr[:, 0].astype(int)
        mask = (n_arr >= n_min) & (n_arr <= n_max)
        rows = arr[mask]
        ns = n_arr[mask]
        for i, n in enumerate(ns):
            key = (p, int(n))
            spectra[key] = rows[i, 1:]

    return spectra

def get_boson_num(n: int) -> int:
    closest_magic = min(CONFIG["nuclei"]["magic_numbers"], key=lambda x: abs(n - x))
    return abs(n - closest_magic) // 2

def get_casten_factor(n_pi: int, n_nu: int) -> float:
    if n_pi + n_nu == 0:
        return 0.0
    # P = (Np * Nn) / (Np + Nn) where N are valence nucleon numbers (= 2 * boson numbers)
    return 2.0 * (n_pi * n_nu) / (n_pi + n_nu)

def _make_split_indices(X: torch.Tensor,
                        val_ratio: float,
                        seed: int) -> tuple:
    """ make train/val split indices """
    rng = np.random.default_rng(seed)

    # X[:, 0] is n_nu (was N)
    group_ids = X[:, 0].detach().cpu().numpy().astype(int)
    unique_ids = np.unique(group_ids)

    idx_train: list[int] = []
    idx_val: list[int] = []

    for gid in unique_ids:
        group_idx = np.where(group_ids == gid)[0]
        group_size = group_idx.size
        if group_size == 0:
            continue
        rng.shuffle(group_idx)

        raw_val_count = int(round(group_size * val_ratio))
        val_count = max(0, min(group_size - 1, raw_val_count))

        if val_count > 0:
            idx_val.extend(group_idx[:val_count].tolist())
        idx_train.extend(group_idx[val_count:].tolist())

    idx_train = torch.tensor(idx_train, dtype=torch.long)
    idx_val = torch.tensor(idx_val, dtype=torch.long)
    return idx_train, idx_val

def minmax_scaler(X: torch.Tensor):
    min_x = X.min(dim=0, keepdim=True).values
    max_x = X.max(dim=0, keepdim=True).values
    range_x = max_x - min_x
    range_x = torch.where(range_x == 0, torch.ones_like(range_x), range_x)
    return min_x, range_x

def apply_minmax_scaler(X: torch.Tensor, min_x: torch.Tensor, range_x: torch.Tensor) -> torch.Tensor:
    return (X - min_x) / range_x



def _prepare_training_dataset(raw_dict: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """ generate training dataset X, Y from raw_dict(N -> array[[beta, E(beta)], ...])
    
    X: (num_samples, 3) = [n_nu, P, beta]
    Y: (num_samples, ) = E(beta) - E(beta=0)
    """
    X_rows: list[list[float]] = []
    Y_vals: list[float] = []
    for (p, n), arr in raw_dict.items():
        if arr is None:
            continue
        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"[WARN] Skipped N={n} : expected 2 columns [beta,E], got shape {arr.shape}")
            continue
        n_nu = get_boson_num(int(n))
        n_pi = get_boson_num(int(p))
        P = get_casten_factor(n_pi, n_nu)
        
        beta_arr = arr[:, 0]
        energies = arr[:, 1]

        # Filter by beta range if specified in config
        if "beta_min" in CONFIG["training"] and "beta_max" in CONFIG["training"]:
            b_min = CONFIG["training"]["beta_min"]
            b_max = CONFIG["training"]["beta_max"]
            mask = (beta_arr >= b_min) & (beta_arr <= b_max)
            beta_arr = beta_arr[mask]
            energies = energies[mask]

        if beta_arr.size == 0:
            print(f"[WARN] No data left for N={n} after beta filtering.")
            continue

        # beta = 0 is 
        idx_beta0 = np.where(np.isclose(beta_arr, 0.0))[0]
        if idx_beta0.size == 0:
            # If exact 0.0 is missing after filtering, try to find it in original array to get E0 reference
            # But E0 must be subtracted. If 0.0 is not in the training range, we should probably still use E(0) from raw data as reference?
            # Let's assume E(0) is always needed for normalization.
            # Re-fetch E0 from original array
            orig_beta = arr[:, 0]
            orig_E = arr[:, 1]
            idx_orig_0 = np.where(np.isclose(orig_beta, 0.0))[0]
            if idx_orig_0.size == 0:
                 raise ValueError(f"No beta=0 point for N={n} (needed for normalization)")
            e0 = orig_E[idx_orig_0[0]]
        else:
            e0 = energies[idx_beta0[0]]
        
        energies -= e0
        n_nu_arr = np.full_like(beta_arr, n_nu)
        n_pi_arr = np.full_like(beta_arr, n_pi)
        P_arr = np.full_like(beta_arr, P, dtype=float)
        X_rows_np = np.stack([n_nu_arr, n_pi_arr, P_arr, beta_arr], axis=1)
        X_rows.extend(X_rows_np.tolist())
        Y_vals.extend(energies.tolist())

    X = np.array(X_rows)
    Y = np.array(Y_vals)
    return X, Y

def _save_training_dataset(X: np.ndarray, Y: np.ndarray, basename: str = "training_dataset") -> dict:
    """ save training data X, Y to .npy and .csv files, return paths """
    processed_dir = CONFIG["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    x_path = processed_dir / f"{basename}_X.npy"
    y_path = processed_dir / f"{basename}_Y.npy"
    np.save(x_path, X)
    np.save(y_path, Y)
    paths["X"] = x_path
    paths["Y"] = y_path

    # also save as CSV for convenience
    csv_path = processed_dir / f"{basename}.csv"
    try:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["n_nu", "n_pi", "P", "beta", "E"])  # header
            for (n_nu, n_pi, P, beta), energy in zip(X, Y):
                writer.writerow([int(n_nu), int(n_pi), float(P), float(beta), f"{float(energy):.3f}"])
        paths["csv"] = csv_path
    except Exception as e:
        print(f"[WARN] Failed to write CSV: {e}")
    return paths

def load_training_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """ load processed data X, Y from .npy files """
    processed_dir = CONFIG["paths"]["processed_dir"]
    x_path = processed_dir / "training_dataset_X.npy"
    y_path = processed_dir / "training_dataset_Y.npy"
    X = torch.from_numpy(np.load(x_path)).float()
    Y = torch.from_numpy(np.load(y_path)).float()
    return X, Y



def _prepare_eval_dataset(raw_data: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """ prepare eval dataset by finding beta min from raw data """
    X_rows: list[list[float]] = []
    for (p, n), arr in raw_data.items():
        if arr is None:
            continue
        n_pi = get_boson_num(int(p))
        n_nu = get_boson_num(int(n))
        P = get_casten_factor(n_pi, n_nu)
        beta_arr = arr[:, 0]
        energies = arr[:, 1]

        idx_beta_min = np.argmin(energies)
        beta_min = beta_arr[idx_beta_min]
        X_rows.append([n, p, n_nu, n_pi, P, beta_min])
    return np.array(X_rows)

def _save_eval_dataset(X_eval: np.ndarray, basename: str) -> dict:
    """ save eval dataset to .npy and csv files, return paths """
    processed_dir = CONFIG["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    npy_path = processed_dir / f"{basename}.npy"
    np.save(npy_path, X_eval)
    paths["npy"] = npy_path

    # also save as CSV for convenience
    csv_path = processed_dir / f"{basename}.csv"
    try:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["N", "Z", "n_nu", "n_pi", "P", "beta_min"])  # header
            for n, p, n_nu, n_pi, P, beta_min in X_eval:
                writer.writerow([int(n), int(p), int(n_nu), int(n_pi), float(P), float(beta_min)])
        paths["csv"] = csv_path
    except Exception as e:
        print(f"[WARN] Failed to write eval inputs CSV: {e}")
    return paths

def load_eval_dataset(basename: str) -> tuple[torch.Tensor, torch.Tensor]:
    """ load eval dataset from .npy file """
    processed_dir = CONFIG["paths"]["processed_dir"]
    npy_path = processed_dir / f"{basename}.npy"
    X_eval = torch.from_numpy(np.load(npy_path)).float()
    
    # Load scaler locally to avoid import-time error
    scaler = load_scaler(CONFIG)
    
    # X_eval has [N, Z, n_nu, n_pi, P, beta_min]
    # Model expects [n_nu, n_pi, P, beta] -> Indices 2, 3, 4, 5
    X_features = X_eval[:, 2:]
    X_eval_scaled = apply_minmax_scaler(X_features, scaler["min"], scaler["range"])
    return X_eval, X_eval_scaled



def main():
    raw_data = load_raw_HFB_energies(
        CONFIG["nuclei"]["p_min"],
        CONFIG["nuclei"]["p_max"],
        CONFIG["nuclei"]["n_min"],
        CONFIG["nuclei"]["n_max"],
        CONFIG["nuclei"]["p_step"],
        CONFIG["nuclei"]["n_step"],
    )
    X, Y = _prepare_training_dataset(raw_data)
    saved_paths = _save_training_dataset(X, Y, "training_dataset")
    print(f"Training data saved: {saved_paths}")
    X_eval = _prepare_eval_dataset(raw_data)
    eval_paths = _save_eval_dataset(X_eval, "eval_dataset")
    print(f"Eval data saved: {eval_paths}")

if __name__ == "__main__":
    main()
