import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from encoder_decoder import EncoderDecoderModel

# parameters (change path, keep the rest)
DATA_DIR = "C:/Users/setoain/OneDrive - Universitat de Barcelona/Escritorio/EEGData"
EPOCHS = 200
eeg_window = 1.05
atlas_parcels = 100
target_epochs = 184
hidden_size = 64

# plotting graphs for paper
def plot_true_vs_pred_arrays(y_true, y_pred, subject, session, region_idx=0):
    y_true_region = y_true[region_idx, :]
    y_pred_region = y_pred[region_idx, :]

    plt.figure(figsize=(12, 5))
    plt.plot(y_true_region, label="True fMRI", color="blue")
    plt.plot(y_pred_region, label="Predicted fMRI", color="red", alpha=0.7)
    plt.title(f"fMRI Time-Series Overlay | {subject} {session} | Region {region_idx}")
    plt.xlabel("Time Points")
    plt.ylabel("Normalized Signal")
    plt.legend()
    plt.show()


def load_eeg_fmri(subject, session, data_dir):
    subject_path = os.path.join(data_dir, subject, session)

    eeg_file = None
    for fname in os.listdir(subject_path):
        if fname.endswith(f"features_table({eeg_window}).csv"):
            eeg_file = os.path.join(subject_path, fname)
            break
    if eeg_file is None:
        raise FileNotFoundError(f"No EEG file ending in 'features_table({eeg_window}).csv' found in {subject_path}")

    fmri_file = os.path.join(
        subject_path,
        f"{subject}_{session}_task-rest_space-MNI152Lin_res-3mm_atlas-Schaefer2018_dens-{atlas_parcels}parcels7networks_desc-sm0_bold.tsv"
    )

    eeg_df = pd.read_csv(eeg_file)
    eeg_data = eeg_df.iloc[:, 1:]
    eeg_data_clean = eeg_data.dropna(how="all")

    eeg = eeg_data_clean.values[:, :target_epochs]
    eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / np.std(eeg, axis=1, keepdims=True)

    with open(fmri_file, "r") as f:
        lines = f.readlines()

    parsed = []
    for line in lines:
        try:
            row = [float(x) for x in line.strip().split('\t')]
            parsed.append(row)
        except ValueError:
            continue

    row_lengths = [len(row) for row in parsed]
    most_common_len = max(set(row_lengths), key=row_lengths.count)
    fmri_clean = [row for row in parsed if len(row) == most_common_len]
    fmri = np.array(fmri_clean, dtype=np.float32)[:, :target_epochs]
    fmri = (fmri - np.mean(fmri, axis=1, keepdims=True)) / np.std(fmri, axis=1, keepdims=True)

    return eeg.astype(np.float32), fmri.astype(np.float32)


def compute_pearson(y_true, y_pred):
    valid_regions = [(np.std(y_true[i]) > 0 and np.std(y_pred[i]) > 0) for i in range(y_true.shape[0])]
    valid_y_true = y_true[valid_regions]
    valid_y_pred = y_pred[valid_regions]

    mean_pearson = np.mean([pearsonr(valid_y_true[i], valid_y_pred[i])[0] for i in range(valid_y_true.shape[0])])
    overall_pearson = pearsonr(y_true.flatten(), y_pred.flatten())[0]

    return mean_pearson, overall_pearson


def train(subject, session, data_dir, epochs=200, lr=1e-3, weight_decay=1e-4, patience=20):
    eeg, fmri = load_eeg_fmri(subject, session, data_dir)

    T = eeg.shape[1]
    train_T = int(0.7 * T)
    val_T = int(0.15 * T)
    test_T = T - train_T - val_T

    train_slice = slice(0, train_T)
    val_slice = slice(train_T, train_T + val_T)
    test_slice = slice(train_T + val_T, T)

    x_full = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0)
    y_full = torch.tensor(fmri, dtype=torch.float32).unsqueeze(0)

    model = EncoderDecoderModel(input_features=eeg.shape[0],
                                hidden_size=hidden_size,
                                output_regions=fmri.shape[0],
                                output_seq_len=T)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output_full = model(x_full, target_output=y_full.unsqueeze(-1), teacher_forcing_ratio=0.5)
        train_loss = criterion(output_full[:, :, train_slice], y_full[:, :, train_slice])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(x_full, target_output=None, teacher_forcing_ratio=0.0)
            val_loss = criterion(val_output[:, :, val_slice], y_full[:, :, val_slice]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        output_full_eval = model(x_full, target_output=None, teacher_forcing_ratio=0.0)
        test_loss = criterion(output_full_eval[:, :, test_slice], y_full[:, :, test_slice]).item()

    y_true_np = y_full[:, :, test_slice].squeeze(0).numpy()
    y_pred_np = output_full_eval[:, :, test_slice].squeeze(0).numpy()

    mean_pearson, overall_pearson = compute_pearson(y_true_np, y_pred_np)

    print(f"{subject} {session} | Train MSE: {train_loss.item():.4f} | Test MSE: {test_loss:.4f}")
    print(f"{subject} {session} | Mean Pearson: {mean_pearson:.4f} | Overall Pearson: {overall_pearson:.4f}")

    return model, test_loss, mean_pearson, overall_pearson


if __name__ == "__main__":
    for subject in sorted(os.listdir(DATA_DIR)):
        subject_path = os.path.join(DATA_DIR, subject)
        if not os.path.isdir(subject_path):
            continue
        for session in sorted(os.listdir(subject_path)):
            session_path = os.path.join(subject_path, session)
            if not os.path.isdir(session_path):
                continue
            try:
                train(subject, session, DATA_DIR, epochs=EPOCHS)
            except Exception as e:
                print(f"[ERROR] {subject} {session}: {e}")
