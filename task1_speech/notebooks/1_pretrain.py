# 1 —Pretraining (Self-Supervised)

## This notebook pretrains an encoder on MEG with random masking (PreTrain-style).
## It produces a checkpoint that is later used to initialize the supervised model.

# Imports are assumed from Notebook 0 if running in one kernel. Re-import key pieces if running standalone.
import os, torch, torch.nn as nn, torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pnpl.datasets import LibriBrainSpeech
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from tqdm import tqdm

def downsample_raw_data(channel_data_tensor, target_len):
    channel_data_tensor = channel_data_tensor.float()
    original_len = channel_data_tensor.size(0)
    if original_len < target_len:
        padding = target_len - original_len
        return F.pad(channel_data_tensor, (0, padding), 'constant', 0.0)
    chunk_size = original_len // target_len
    processed_len = chunk_size * target_len
    trimmed = channel_data_tensor[:processed_len]
    return trimmed.view(target_len, chunk_size).mean(dim=1)

def apply_ssp_masking(input_embedding_tensor: torch.Tensor, mask_ratio: float = 0.6):
    B, D_e, T_reduced = input_embedding_tensor.shape
    num_mask = int(T_reduced * mask_ratio)
    if num_mask == 0:
        return input_embedding_tensor.clone(), torch.empty(B, 0, D_e, device=input_embedding_tensor.device), [], [torch.empty(0, dtype=torch.long) for _ in range(B)]
    if num_mask >= T_reduced:
        original_masked_features = input_embedding_tensor.permute(0, 2, 1).clone()
        masked_input_embedding = torch.zeros_like(input_embedding_tensor)
        masked_indices_list = [torch.arange(T_reduced, device=input_embedding_tensor.device) for _ in range(B)]
        return masked_input_embedding, original_masked_features, [], masked_indices_list

    masked = input_embedding_tensor.clone()
    masked_indices_list, context_indices_list, original_masked_features_list = [], [], []
    for i in range(B):
        perm = torch.randperm(T_reduced, device=input_embedding_tensor.device)
        m_idx = torch.sort(perm[:num_mask]).values
        c_idx = torch.sort(perm[num_mask:]).values
        masked_indices_list.append(m_idx); context_indices_list.append(c_idx)
        original_masked_features_list.append(input_embedding_tensor[i, :, m_idx].permute(1, 0))
        masked[i, :, m_idx] = 0.0
    original_masked_features = torch.stack(original_masked_features_list, dim=0)
    return masked, original_masked_features, context_indices_list, masked_indices_list

class PreTrainInputEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim, reduced_time_steps):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_dim), nn.GELU(),
            nn.AdaptiveAvgPool1d(reduced_time_steps)
        )
    def forward(self, x): return self.conv_layers(x)

class PreTrainTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=4*embedding_dim, dropout=dropout,
            activation=F.gelu, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
    def forward(self, x): return self.enc(self.norm(x))

class PreTrainPredictor(nn.Module):
    def __init__(self, embedding_dim, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(embedding_dim, embedding_dim)]
            if i < num_layers - 1: layers += [nn.GELU(), nn.LayerNorm(embedding_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class PreTrainPretrainer(L.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.input_encoder = PreTrainInputEncoder(NUM_TOTAL_MEG_CHANNELS, hparams['PreTrain_embedding_dim'], hparams['PreTrain_reduced_time_steps'])
        self.context_transformer =PreTrainTransformerEncoder(hparams['PreTrain_embedding_dim'], hparams['PreTrain_transformer_heads'], hparams['PreTrain_transformer_layers'], DROPOUT_RATE)
        self.target_transformer  = PreTrainTransformerEncoder(hparams['PreTrain_embedding_dim'], hparams['PreTrain_transformer_heads'], hparams['PreTrain_transformer_layers'], DROPOUT_RATE)
        self.target_transformer.load_state_dict(self.context_transformer.state_dict())
        for p in self.target_transformer.parameters(): p.requires_grad = False
        self.predictor = PreTrainPredictor(hparams['PreTrain_embedding_dim'], num_layers=PreTrain_PREDICTOR_DEPTH)
        self.mse = nn.MSELoss()
    def forward(self, x):
        init = self.input_encoder(x)                           # (B, D, T_r)
        init_tr = init.permute(0,2,1)                          # (B, T_r, D)
        masked, _, _, masked_idx = apply_ssp_masking(init, PreTrain_MASK_RATIO)
        ctx = self.context_transformer(masked.permute(0,2,1))  # (B, T_r, D)
        valid = [i for i in range(x.size(0)) if masked_idx[i].numel()>0]
        if not valid:
            return torch.empty(0, self.hparams['PreTrain_embedding_dim'], device=x.device), torch.empty(0, self.hparams['PreTrain_embedding_dim'], device=x.device)
        ctx_at_mask = torch.cat([ctx[i, masked_idx[i], :] for i in valid], dim=0)
        with torch.no_grad():
            tgt_full = self.target_transformer(init_tr)
            tgt_at_mask = torch.cat([tgt_full[i, masked_idx[i], :] for i in valid], dim=0)
        pred = self.predictor(ctx_at_mask)
        return pred, tgt_at_mask
    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        pred, tgt = self(x)
        loss = self.mse(pred, tgt) if pred.numel() > 0 else torch.tensor(0.0, device=self.device)
        self.log("pretrain_loss", loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=PRETRAIN_LEARNING_RATE, weight_decay=1e-2)
        sch = CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

  # Example keys mirror your script (modify as needed)
train_keys = [( "0", f"{i}", "Sherlock2", "1" ) for i in range(1,13) if i != 2] \               + [( "0", f"{i}", "Sherlock4", "1" ) for i in range(1,13) if i != 8] \               + [( "0", f"{i}", "Sherlock5", "1" ) for i in range(1,13)] \               + [( "0", f"{i}", "Sherlock6", "1" ) for i in range(2,13)]

class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, is_train=True,
                 apply_gaussian_noise=True, noise_std=0.015,
                 aug_random_crop_min_ratio=0.8, aug_random_crop_max_ratio=1.0,
                 aug_time_warp_max_stretch=0.05, aug_channel_mask_ratio=0.2):
        super().__init__()
        self.ds = dataset
        self.idx = list(range(len(dataset)))
        self.is_train = is_train
        self.apply_gaussian_noise = apply_gaussian_noise
        self.noise_std = noise_std
        self.aug_random_crop_min_ratio = aug_random_crop_min_ratio
        self.aug_random_crop_max_ratio = aug_random_crop_max_ratio
        self.aug_time_warp_max_stretch = aug_time_warp_max_stretch
        self.aug_channel_mask_ratio = aug_channel_mask_ratio
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        x, y = self.ds[self.idx[i]]
        x = x.cpu().float(); y = y.cpu().float()
        # length normalize to SEGMENT_TIME_LEN_SAMPLES
        cur = x.shape[1]
        if cur < SEGMENT_TIME_LEN_SAMPLES:
            x = F.pad(x, (0, SEGMENT_TIME_LEN_SAMPLES - cur))
            y = F.pad(y, (0, SEGMENT_TIME_LEN_SAMPLES - cur))
        elif cur > SEGMENT_TIME_LEN_SAMPLES:
            x = x[:, :SEGMENT_TIME_LEN_SAMPLES]; y = y[:SEGMENT_TIME_LEN_SAMPLES]
        # light augmentations (optional for pretrain)
        if self.is_train and self.apply_gaussian_noise:
            x = x + torch.randn_like(x) * self.noise_std
        return x, torch.empty(0), y  # keep tuple shape compatible

unlabeled = LibriBrainSpeech(data_path=f"{BASE_PATH}/data/", include_run_keys=train_keys,
                             tmin=0.0, tmax=4.0, preload_files=False, standardize=True,
                             oversample_silence_jitter=0)
pretrain_ds = FilteredDataset(unlabeled, is_train=True)
pretrain_loader = DataLoader(pretrain_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

hparams = dict(PreTrain_embedding_dimPreTrain_EMBEDDING_DIM,
               PreTrain_reduced_time_steps=PreTrain_REDUCED_TIME_STEPS,
               PreTrain_transformer_heads=PreTrain_TRANSFORMER_HEADS,
               PreTrainp_transformer_layers=PreTrain_TRANSFORMER_LAYERS)

model = PreTrainPretrainer(**hparams)
logger = CSVLogger(save_dir=f"{BASE_PATH}/lightning_logs", name="", version="PreTrain_pretrain_run")
ckpt_cb = ModelCheckpoint(dirpath=f"{BASE_PATH}/models", filename="PreTrain_pretrain_best_{epoch:02d}-{pretrain_loss:.3f}", monitor="pretrain_loss", mode="min", save_top_k=1)
trainer = L.Trainer(devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu", max_epochs=PRETRAIN_EPOCHS, logger=logger, callbacks=[ckpt_cb, TQDMProgressBar(refresh_rate=1)])
print("Ready to pretrain. (Skip execution here if you just wanted the notebook files.)")

