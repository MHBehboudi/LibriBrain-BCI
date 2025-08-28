# 4 — Ensemble Weight Search & Submission Generation
# Optimizes alpha for two-model ensemble and writes the submission CSV for the competition holdout.

import os
import numpy as np
import pandas as pd
import torch
import lightning as L
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# --- USER INPUT: set your two fine-tuned checkpoints here ---
SPEECH_CKPT   = os.path.join(BASE_PATH, "models", "best_fine_tuned_eeg2rep_hybrid_speech_focused.ckpt")
SILENCE_CKPT  = os.path.join(BASE_PATH, "models", "best_fine_tuned_eeg2rep_hybrid_silence_focused.ckpt")

# Optional: if you saved optimal thresholds during training into hparams, they'll be loaded automatically.

# --- Build validation loader for probability collection (uses your val dataset) ---
# We reuse the same val split as in training.
val_keys = [("0","11","Sherlock1","2")]
val_raw   = LibriBrainSpeech(
    data_path=f"{BASE_PATH}/data/",
    include_run_keys=val_keys,
    tmin=0.0, tmax=4.0,
    preload_files=False, standardize=True, oversample_silence_jitter=0
)

# This FilteredDataset should match the implementation used during validation in your training notebook.
class _FilteredValDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.ds = dataset
        self.idx = list(range(len(dataset)))
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        x, y = self.ds[self.idx[i]]
        x = x.cpu().float(); y = y.cpu().float()
        # length normalize
        cur = x.shape[1]
        if cur < SEGMENT_TIME_LEN_SAMPLES:
            x = torch.nn.functional.pad(x, (0, SEGMENT_TIME_LEN_SAMPLES - cur))
            y = torch.nn.functional.pad(y, (0, SEGMENT_TIME_LEN_SAMPLES - cur))
        elif cur > SEGMENT_TIME_LEN_SAMPLES:
            x = x[:, :SEGMENT_TIME_LEN_SAMPLES]; y = y[:SEGMENT_TIME_LEN_SAMPLES]
        # Build minimal GNN features (match your training path — clusters optional)
        gnn_feats = []
        x_masked = x[SENSORS_SPEECH_MASK].float()
        for ch in range(len(SENSORS_SPEECH_MASK)):
            sig = x_masked[ch, :]
            # reuse helper if available; otherwise inline downsample
            chunk = max(1, sig.numel() // RAW_DOWNSAMPLE_LEN)
            proc_len = chunk * RAW_DOWNSAMPLE_LEN
            sig_trim = sig[:proc_len].view(RAW_DOWNSAMPLE_LEN, chunk).mean(dim=1)
            pad = torch.nn.functional.pad(sig_trim, (0, max(0, GNN_INPUT_NODE_FEATURE_COMMON_DIM - RAW_DOWNSAMPLE_LEN)))
            gnn_feats.append(pad)
        # fill cluster nodes with zeros if you didn't precompute them here
        for _ in range(NUM_GNN_SPATIAL_CLUSTER_NODES):
            gnn_feats.append(torch.zeros(GNN_INPUT_NODE_FEATURE_COMMON_DIM))
        gnn_feats = torch.stack(gnn_feats) if gnn_feats else torch.empty(0)
        return x, gnn_feats, y

val_filtered = _FilteredValDataset(val_raw)
val_pred_ds  = ValidationPredictionDataset(val_filtered)  # from your notebook 3
val_loader   = DataLoader(val_pred_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

# --- Helper to collect probs from a checkpoint ---
def collect_probs(ckpt_path):
    # Load model; pass minimal required hparams that Lightning expects
    model = SpeechClassifier.load_from_checkpoint(  # from your notebook 2
        checkpoint_path=ckpt_path,
        pos_weight=1.0,  # not used in predict
        batch_size=BATCH_SIZE
    )
    trainer = L.Trainer(devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu", logger=False)
    preds = trainer.predict(model, dataloaders=val_loader)
    # preds are tuples: (probs_at_center, label)
    all_probs, all_labels = [], []
    for p_batch, y_batch in preds:
        all_probs.append(p_batch.detach().cpu().numpy())
        all_labels.append(y_batch.detach().cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)

# --- Collect validation probs for each model ---
speech_val_probs, val_labels = collect_probs(SPEECH_CKPT)
silence_val_probs, _         = collect_probs(SILENCE_CKPT)
assert speech_val_probs.shape == silence_val_probs.shape == val_labels.shape

# --- Grid-search alpha to maximize macro-F1 on validation ---
alphas = np.linspace(0.0, 1.0, 51)
best_alpha, best_macro_f1, best_thresh = 0.5, -1.0, 0.5

for alpha in alphas:
    comb = alpha * speech_val_probs + (1.0 - alpha) * silence_val_probs
    # sweep thresholds
    local_best_f1, local_best_t = -1.0, 0.5
    for t in np.linspace(0.01, 0.99, 100):
        pred = (comb >= t).astype(int)
        f1_pos = f1_score(val_labels, pred, pos_label=1, average='binary', zero_division=0)
        f1_neg = f1_score(val_labels, pred, pos_label=0, average='binary', zero_division=0)
        f1_macro = 0.5 * (f1_pos + f1_neg)
        if f1_macro > local_best_f1:
            local_best_f1, local_best_t = f1_macro, t
    if local_best_f1 > best_macro_f1:
        best_macro_f1, best_alpha, best_thresh = local_best_f1, alpha, local_best_t

print(f"[Ensemble] Best alpha: {best_alpha:.3f} | Best macro-F1 (val): {best_macro_f1:.4f} | Threshold: {best_thresh:.3f}")

# --- Build holdout loader (sliding-window dataset) ---
holdout = LibriBrainCompetitionHoldout(data_path=f"{BASE_PATH}/data/", tmax=4.0, task="speech")
sw_holdout_ds = SlidingWindowPredictionDataset(holdout, SENSORS_SPEECH_MASK)  # from your notebook 3
holdout_loader = DataLoader(sw_holdout_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

def collect_holdout_probs(ckpt_path):
    model = SpeechClassifier.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        pos_weight=1.0,  # not used in predict
        batch_size=BATCH_SIZE
    )
    trainer = L.Trainer(devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu", logger=False)
    probs_list = []
    idx_list = []
    for p_batch, idx_batch in trainer.predict(model, dataloaders=holdout_loader):
        probs_list.append(p_batch.detach().cpu().numpy().astype(np.float32))
        idx_list.append(idx_batch.detach().cpu().numpy())
    probs = np.concatenate(probs_list) if probs_list else np.array([])
    indices = np.concatenate(idx_list) if idx_list else np.array([], dtype=int)
    # reshape to full-length vector over the base holdout timeline
    full = np.zeros(len(holdout), dtype=np.float32)
    full[indices] = probs
    return full

speech_holdout = collect_holdout_probs(SPEECH_CKPT)
silence_holdout = collect_holdout_probs(SILENCE_CKPT)
assert speech_holdout.shape == silence_holdout.shape == (len(holdout),)

# --- Combine with the optimized alpha and threshold to make binary predictions ---
combined_holdout = best_alpha * speech_holdout + (1.0 - best_alpha) * silence_holdout
final_binary = (combined_holdout >= best_thresh).astype(np.int32)

# --- Write submission ---
os.makedirs(SUBMISSION_PATH, exist_ok=True)
out_csv = os.path.join(SUBMISSION_PATH, "libribrain_speech_submission_eeg2rep_hybrid_ensemble_optimized.csv")
pd.DataFrame({"speech_prob": final_binary}).to_csv(out_csv, index=False)
print(f"[✓] Wrote submission: {out_csv}")
