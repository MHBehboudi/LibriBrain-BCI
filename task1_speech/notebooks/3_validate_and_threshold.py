# This notebook loads checkpoints, computes validation probabilities,
# and searches thresholds maximizing macro-F1.

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning as L
from sklearn.metrics import f1_score


class ValidationPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, base_val_filtered_dataset,
                 use_seq2seq: bool = None,
                 reduced_steps: int = None):
        self.base = base_val_filtered_dataset
        self.use_seq2seq = USE_SEQ2SEQ_PREDICTION if use_seq2seq is None else use_seq2seq
        self.red_steps = EEG2REP_REDUCED_TIME_STEPS if reduced_steps is None else reduced_steps

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        # Expecting base dataset returns: (raw_meg_full_306ch, gnn_input_features, labels_seq_processed)
        x, g, y_seq = self.base[i]

        if self.use_seq2seq:
            # Downsample labels to model's output steps and take the center timepoint
            y_ds = F.adaptive_avg_pool1d(y_seq.unsqueeze(0).unsqueeze(0).float(),
                                         self.red_steps).squeeze(0).squeeze(0)
            label = y_ds[self.red_steps // 2].item()
        else:
            # Non-seq2seq: aggregate to a single label (mean over segment)
            label = y_seq.float().mean().item()

        # Return label as int 0/1 (long). If you prefer floats, switch to dtype=torch.float32.
        return x, g, torch.tensor(int(round(label)), dtype=torch.long)
