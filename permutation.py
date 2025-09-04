# permutation_groups_auto.py
import os
import re
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional

from encoder_decoder import EncoderDecoderModel

# parameters (choose subject/session to conduct, also change path if necessary, leabe the rest)
SUBJECT      = "sub-01"
SESSION      = "ses-01"

EEG_DATA_DIR = r"C:/Users/setoain/OneDrive - Universitat de Barcelona/Escritorio/EEGData"
MODEL_DIR    = r"C:/Users/setoain/OneDrive - Universitat de Barcelona/Escritorio/eeg2fmri-project/models"
OUTPUT_DIR   = r"C:/Users/setoain/OneDrive - Universitat de Barcelona/Escritorio/eeg2fmri-project/outputs"

eeg_window   = 1.05
hidden_size  = 64
n_perm       = 10
seed         = 0

FAKE_PREFIX  = ""   # optional prefix


def load_array(path_no_ext: str) -> np.ndarray:
    if os.path.exists(path_no_ext + ".npy"):
        return np.load(path_no_ext + ".npy")
    if os.path.exists(path_no_ext + ".csv"):
        return np.loadtxt(path_no_ext + ".csv", delimiter=",")
    raise FileNotFoundError(f"Missing: {path_no_ext}.npy or .csv")

def load_y_true(subject: str, session: str) -> np.ndarray:
    base = os.path.join(OUTPUT_DIR, f"{subject}_{session}_true_fmri_test")
    return load_array(base)

def pearson_flattened(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    a = y_true.reshape(-1).astype(np.float64)
    b = y_pred.reshape(-1).astype(np.float64)
    a -= a.mean(); b -= b.mean()
    denom = np.sqrt((a*a).sum()) * np.sqrt((b*b).sum()) + eps
    return float((a @ b) / denom)

def mse_flattened(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_true.reshape(-1).astype(np.float64) - y_pred.reshape(-1).astype(np.float64)
    return float(np.mean(d * d))

def find_eeg_file(subject: str, session: str) -> str:
    subj_path = os.path.join(EEG_DATA_DIR, subject, session)
    base = f"features_table({eeg_window}).csv"
    if FAKE_PREFIX:
        pref = os.path.join(subj_path, FAKE_PREFIX + base)
        if os.path.exists(pref):
            return pref
    orig = os.path.join(subj_path, base)
    if os.path.exists(orig):
        return orig
    for fn in os.listdir(subj_path):
        if fn.endswith(base) or (FAKE_PREFIX and fn.endswith(FAKE_PREFIX + base)):
            return os.path.join(subj_path, fn)
    raise FileNotFoundError(f"EEG CSV not found in {subj_path} (looked for {FAKE_PREFIX+base} or {base})")

def load_clean_eeg_df(eeg_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(eeg_csv_path)
    df = df.dropna(how="all")
    epoch_cols = [c for c in df.columns if c.lower().startswith("epoch_")]
    if epoch_cols:
        df = df.loc[~df[epoch_cols].isna().all(axis=1)]
    return df.reset_index(drop=True)

def load_training_feature_list(subject: str, session: str) -> Optional[List[str]]:
    feat_path = os.path.join(MODEL_DIR, f"{subject}_{session}_features.txt")
    if os.path.exists(feat_path):
        with open(feat_path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]
    return None

def expected_F_from_checkpoint(ckpt_path: str, hidden_size: int) -> int:
    state = torch.load(ckpt_path, map_location="cpu")
    keys = [
        "encoder.lstm.weight_ih_l0",
        "lstm.weight_ih_l0",
        "encoder.rnn.weight_ih_l0"
    ]
    for k in keys:
        if k in state:
            return int(state[k].shape[1])
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and "weight_ih_l0" in k:
            return int(v.shape[1])
    raise RuntimeError("Could not infer input feature count from checkpoint.")

def align_x(
    df: pd.DataFrame,
    train_names: Optional[List[str]],
    Ttest: int,
    expected_F: int
) -> Tuple[np.ndarray, List[str]]:
    if "Feature" not in df.columns:
        raise ValueError("Expected a 'Feature' column with feature names.")
    feature_names = df["Feature"].astype(str).tolist()
    epoch_cols = [c for c in df.columns if c.lower().startswith("epoch_")]
    if not epoch_cols:
        epoch_cols = [c for c in df.columns if c != "Feature"]
    data = df[epoch_cols].to_numpy(dtype=np.float32)
    if data.shape[1] < Ttest:
        raise ValueError(f"EEG has only {data.shape[1]} timepoints; need Ttest={Ttest}")
    data = data[:, -Ttest:]
    mu = data.mean(axis=1, keepdims=True)
    sd = data.std(axis=1, keepdims=True) + 1e-8
    data = (data - mu) / sd
    F_curr = data.shape[0]

    if train_names is not None:
        idx_map = {nm: i for i, nm in enumerate(feature_names)}
        x = np.zeros((len(train_names), Ttest), dtype=np.float32)
        for j, nm in enumerate(train_names):
            i = idx_map.get(nm, None)
            if i is not None:
                x[j, :] = data[i, :]
        if x.shape[0] != expected_F:
            if x.shape[0] < expected_F:
                pad = expected_F - x.shape[0]
                x = np.vstack([x, np.zeros((pad, Ttest), dtype=np.float32)])
                train_names = train_names + [f"MISSING_{k}" for k in range(pad)]
            else:
                x = x[:expected_F, :]
                train_names = train_names[:expected_F]
        return x, train_names

    names = feature_names.copy()
    x = data.copy()
    if F_curr < expected_F:
        pad = expected_F - F_curr
        x = np.vstack([x, np.zeros((pad, Ttest), dtype=np.float32)])
        names += [f"MISSING_{k}" for k in range(pad)]
    elif F_curr > expected_F:
        x = x[:expected_F, :]
        names = names[:expected_F]
    return x, names

def build_and_load_model(subject: str, session: str, F: int, R: int, Ttest: int, ckpt_path: str):
    model = EncoderDecoderModel(
        input_features=F,
        hidden_size=hidden_size,
        output_regions=R,
        output_seq_len=Ttest
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# feature type grouping
def build_groups(feature_names: List[str]) -> Dict[str, List[int]]:
    names = feature_names
    groups: Dict[str, List[int]] = {}
    rx_band_power = re.compile(r"(delta|theta|alpha|beta|gamma).*amp", re.IGNORECASE)
    groups["band_power_metrics"] = [i for i, n in enumerate(names) if rx_band_power.search(n)]
    rx_peak = re.compile(r"peak\s*freq", re.IGNORECASE)
    groups["peak_frequency"] = [i for i, n in enumerate(names) if rx_peak.search(n)]
    rx_aper = re.compile(r"(1/f|aperiodic).*slope|1/f\s*slope", re.IGNORECASE)
    groups["aperiodic_1f_slope"] = [i for i, n in enumerate(names) if rx_aper.search(n)]
    rx_burst_count = re.compile(r"burst\s*(count|rate)", re.IGNORECASE)
    rx_burst_dur   = re.compile(r"burst\s*duration", re.IGNORECASE)
    groups["burst_rate"]     = [i for i, n in enumerate(names) if rx_burst_count.search(n)]
    groups["burst_duration"] = [i for i, n in enumerate(names) if rx_burst_dur.search(n)]
    rx_plv = re.compile(r"\bplv\b", re.IGNORECASE)
    groups["plv"] = [i for i, n in enumerate(names) if rx_plv.search(n)]
    rx_aec = re.compile(r"\baec\b", re.IGNORECASE)
    groups["aec"] = [i for i, n in enumerate(names) if rx_aec.search(n)]
    rx_gfp = re.compile(r"(global\s*field\s*power|^gfp$)", re.IGNORECASE)
    groups["global_field_power"] = [i for i, n in enumerate(names) if rx_gfp.search(n)]
    return {k: v for k, v in groups.items() if len(v) > 0}


def main():
    subject, session = SUBJECT, SESSION
    y_true = load_y_true(subject, session)
    R, Ttest = y_true.shape
    eeg_csv = find_eeg_file(subject, session)
    df = load_clean_eeg_df(eeg_csv)
    ckpt_path = os.path.join(MODEL_DIR, f"{subject}_{session}_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    expected_F = expected_F_from_checkpoint(ckpt_path, hidden_size)
    train_names = load_training_feature_list(subject, session)
    x_test, feat_names = align_x(df, train_names, Ttest, expected_F)
    model = build_and_load_model(subject, session, x_test.shape[0], R, Ttest, ckpt_path)

    with torch.no_grad():
        y_hat_base = model(torch.tensor(x_test, dtype=torch.float32).unsqueeze(0),
                           target_output=None, teacher_forcing_ratio=0.0).squeeze(0).numpy()
    base_r = pearson_flattened(y_true, y_hat_base)
    base_mse = mse_flattened(y_true, y_hat_base)

    print(f"[{subject} | {session}] Baseline: r={base_r:.4f}, MSE={base_mse:.6f}  (F={x_test.shape[0]}, Ttest={Ttest})")

    groups = build_groups(feat_names)
    if not groups:
        return

    rng = np.random.default_rng(seed)
    rows = []
    for gname, idxs in groups.items():
        diffs_r, diffs_mse = [], []
        for _ in range(n_perm):
            x_pert = x_test.copy()
            perms = [rng.permutation(Ttest) for _ in idxs]
            for f, perm in zip(idxs, perms):
                x_pert[f, :] = x_pert[f, perm]
            with torch.no_grad():
                y_hat = model(torch.tensor(x_pert, dtype=torch.float32).unsqueeze(0),
                              target_output=None, teacher_forcing_ratio=0.0).squeeze(0).numpy()
            r = pearson_flattened(y_true, y_hat)
            mse = mse_flattened(y_true, y_hat)
            diffs_r.append(base_r - r)
            diffs_mse.append(mse - base_mse)
        rows.append((gname, float(np.mean(diffs_r)), float(np.mean(diffs_mse)), len(idxs)))

    print("\nPermutation drops per group (higher = more important):")
    for g, dr, dmse, sz in sorted(rows, key=lambda x: -x[1]):
        print(f"{g:22s}  Δr={dr:.5f}  ΔMSE={dmse:.6f}  (features={sz})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, f"{subject}_{session}_GROUP_perm_drops.csv")
    pd.DataFrame(rows, columns=["group", "delta_pearson", "delta_mse", "group_size"]).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
