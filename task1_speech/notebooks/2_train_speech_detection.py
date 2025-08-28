# 2 — Supervised Training: Speech vs Silence
## This notebook fine-tunes the model for **Task 1 (Speech Detection)** and logs validation metrics.

import torch, torch.nn as nn, torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, AUROC
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, StochasticWeightAveraging
from lightning.pytorch.loggers import CSVLogger
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from pnpl.datasets import LibriBrainSpeech
import numpy as np

## Losses & Attentions
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=3, reduction='mean', pos_weight=None):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float)
        self.register_buffer('pos_weight', pos_weight)
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - pt) ** self.gamma) * bce
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum':  return loss.sum()
        return loss

class ChannelSelfAttention(nn.Module):
    def __init__(self, channels, attn_dim=128, dropout=0.1):
        super().__init__()
        self.d = attn_dim; self.dropout = nn.Dropout(dropout)
        self.q = nn.Linear(1, attn_dim); self.k = nn.Linear(1, attn_dim); self.v = nn.Linear(1, attn_dim)
    def forward(self, x):
        B,C,T = x.size()
        x = x.permute(0,2,1).unsqueeze(-1) # (B,T,C,1)
        Q,K,V = [proj(x) for proj in (self.q,self.k,self.v)]
        Q,K,V = [t.reshape(B*T, C, self.d) for t in (Q,K,V)]
        scores = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(self.d)
        wts = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.bmm(wts, V).mean(dim=1).view(B,T,self.d)
        return out, wts.view(B,T,C,C)

class FusionAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dim_per_head, dropout=0.1):
        super().__init__()
        self.h = num_heads; self.dh = dim_per_head; self.D = num_heads * dim_per_head
        self.qkv = nn.Linear(input_dim, 3*self.D)
        self.out = nn.Linear(self.D, input_dim)
        self.drop = nn.Dropout(dropout); self.norm = nn.LayerNorm(input_dim)
    def forward(self, x):
        r = x; x = self.norm(x)
        B,D = x.size()
        qkv = self.qkv(x).reshape(B,3,self.h,self.dh)
        q,k,v = qkv.unbind(dim=1)
        q = q.transpose(0,1).reshape(self.h*B,1,self.dh)
        k = k.transpose(0,1).reshape(self.h*B,1,self.dh)
        v = v.transpose(0,1).reshape(self.h*B,1,self.dh)
        att = torch.softmax(torch.bmm(q,k.transpose(1,2))/math.sqrt(self.dh), dim=-1)
        out = torch.bmm(att,v).reshape(self.h,B,self.dh).transpose(0,1).reshape(B,self.D)
        return r + self.drop(self.out(out))

  class EEG2RepInputEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim, reduced_time_steps):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2,2),
            nn.Conv1d(128, embedding_dim, 3, padding=1), nn.BatchNorm1d(embedding_dim), nn.GELU(),
            nn.AdaptiveAvgPool1d(reduced_time_steps)
        )
    def forward(self, x): return self.conv(x)

## Core Model
class EEG2RepTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, heads, layers, dropout):
        super().__init__()
        lyr = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=heads, dim_feedforward=4*embedding_dim, dropout=dropout, activation='gelu', batch_first=True)
        self.enc = nn.TransformerEncoder(lyr, num_layers=layers)
        self.norm = nn.LayerNorm(embedding_dim)
    def forward(self, x): return self.enc(self.norm(x))


class SpeechModelHybridGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg_in = EEG2RepInputEncoder(NUM_TOTAL_MEG_CHANNELS, EEG2REP_EMBEDDING_DIM, EEG2REP_REDUCED_TIME_STEPS)
        self.eeg_enc = EEG2RepTransformerEncoder(EEG2REP_EMBEDDING_DIM, EEG2REP_TRANSFORMER_HEADS, EEG2REP_TRANSFORMER_LAYERS, DROPOUT_RATE)

        self.cnn_in = NUM_RAW_CHANNELS_CNN_STREAM
        self.conv = nn.Conv1d(self.cnn_in, CONV_DIM, 3, padding=1)
        self.bn = nn.BatchNorm1d(CONV_DIM)
        self.conv_drop = nn.Dropout(DROPOUT_RATE)
        self.chan_attn = ChannelSelfAttention(self.cnn_in, ATTN_DIM, DROPOUT_RATE)
        self.tf_cnn_lyr = nn.TransformerEncoderLayer(d_model=CONV_DIM+ATTN_DIM, nhead=TRANSFORMER_NHEAD, dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD, dropout=DROPOUT_RATE, activation='relu', batch_first=True)
        self.tf_cnn = nn.TransformerEncoder(self.tf_cnn_lyr, num_layers=CNN_ATTN_TRANSFORMER_LAYERS)
        self.cnn_res_proj = nn.Conv1d(self.cnn_in, CONV_DIM, 1) if (USE_CNN_RESIDUAL and self.cnn_in != CONV_DIM) else nn.Identity()

        if USE_GNN:
            self.gnn_proj = nn.Linear(GNN_INPUT_NODE_FEATURE_COMMON_DIM, GNN_OUTPUT_DIM)
            self.gnn_layers = nn.ModuleList()
            g_in = GNN_OUTPUT_DIM
            for i in range(NUM_GNN_LAYERS):
                if USE_GAT:
                    heads = 4 if i < NUM_GNN_LAYERS - 1 else 1
                    self.gnn_layers.append(GATConv(g_in, GNN_OUTPUT_DIM, heads=heads, dropout=DROPOUT_RATE, add_self_loops=True))
                    g_in = GNN_OUTPUT_DIM * heads
                else:
                    self.gnn_layers.append(GCNConv(g_in, GNN_OUTPUT_DIM, add_self_loops=True))
                    g_in = GNN_OUTPUT_DIM
            self.gnn_final_dim = g_in
            self.gnn_drop = nn.Dropout(DROPOUT_RATE)
            self.gnn_res = nn.Linear(GNN_INPUT_NODE_FEATURE_COMMON_DIM, self.gnn_final_dim) if GNN_INPUT_NODE_FEATURE_COMMON_DIM != self.gnn_final_dim else nn.Identity()
        else:
            self.gnn_final_dim = 0

        fused_cnn_attn_dim = CONV_DIM + ATTN_DIM
        fusion_in = EEG2REP_EMBEDDING_DIM + fused_cnn_attn_dim + self.gnn_final_dim
        self.fusion_attn = FusionAttention(fusion_in, FUSION_ATTN_HEADS, FUSION_ATTN_DIM_PER_HEAD, DROPOUT_RATE) if USE_FUSION_ATTENTION else nn.Identity()
        self.lstm = nn.LSTM(input_size=fusion_in, hidden_size=fusion_in, num_layers=LSTM_LAYERS, batch_first=True, dropout=DROPOUT_RATE if LSTM_LAYERS>1 else 0, bidirectional=BI_DIRECTIONAL)
        self.lstm_drop = nn.Dropout(DROPOUT_RATE)
        clf_in = fusion_in * (2 if BI_DIRECTIONAL else 1)
        self.classifier = nn.Linear(clf_in, 1)

    def forward(self, raw_meg_full_306ch, x_gnn_input_features):
        B,_,_ = raw_meg_full_306ch.size()
        eeg = self.eeg_in(raw_meg_full_306ch).permute(0,2,1)
        eeg = self.eeg_enc(eeg)
        T = eeg.size(1)

        x23 = raw_meg_full_306ch[:, SENSORS_SPEECH_MASK, :]
        cnn = self.conv(x23)
        if USE_CNN_RESIDUAL:
            res = x23 if x23.shape[1]==self.conv.out_channels else self.cnn_res_proj(x23)
            cnn = cnn + res
        cnn = self.conv_drop(self.bn(cnn)).permute(0,2,1)
        attn_f, attn_w = self.chan_attn(x23)
        attn_f = F.adaptive_avg_pool1d(attn_f.permute(0,2,1), T).permute(0,2,1)
        cnn = F.adaptive_avg_pool1d(cnn.permute(0,2,1), T).permute(0,2,1)
        cnn_attn = torch.cat([cnn, attn_f], dim=-1)
        cnn_attn = self.tf_cnn(cnn_attn)

        gnn_seq = torch.zeros(B, 0, device=raw_meg_full_306ch.device)
        if USE_GNN and x_gnn_input_features.numel()>0:
            proj = self.gnn_proj(x_gnn_input_features)
            node_feats = proj.view(B*NUM_GNN_NODES_TOTAL, -1)
            batch_edges = []
            for i in range(B):
                batch_edges.append(STATIC_GNN_EDGE_INDEX.to(raw_meg_full_306ch.device) + i*NUM_GNN_NODES_TOTAL)
            edges = torch.cat(batch_edges, dim=1)
            g = node_feats
            for i,layer in enumerate(self.gnn_layers):
                g = layer(g, edges)
                if i < NUM_GNN_LAYERS-1:
                    g = F.relu(g); g = self.gnn_drop(g)
            g = F.relu(g)
            if USE_GNN_RESIDUAL:
                r = self.gnn_res(x_gnn_input_features).view(B*NUM_GNN_NODES_TOTAL, -1)
                g = g + r
            g = g.view(B, NUM_GNN_NODES_TOTAL, -1).mean(dim=1)
            gnn_seq = g
        if gnn_seq.numel()==0:
            gnn_seq = torch.zeros(B, 0, device=raw_meg_full_306ch.device)

        gnn_seq = gnn_seq.unsqueeze(1).repeat(1, T, 1)
        fused = torch.cat([eeg, cnn_attn, gnn_seq], dim=-1)
        if USE_SEQ2SEQ_PREDICTION:
            flat = fused.view(B*T, -1)
            flat = self.fusion_attn(flat)
            lstm_in = flat.view(B, T, -1)
            out,_ = self.lstm(lstm_in); out = self.lstm_drop(out)
            logits = self.classifier(out).squeeze(-1)
            return logits, attn_w
        else:
            pooled = fused.mean(dim=1)
            pooled = self.fusion_attn(pooled)
            out,_ = self.lstm(pooled.unsqueeze(1))
            h = out[:,-1,:]
            logits = self.classifier(self.lstm_drop(h)).squeeze(-1)
            return logits, attn_w

## Lightning Wrapper & Training Loop
class SpeechClassifier(L.LightningModule):
    def __init__(self, pos_weight, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = SpeechModelHybridGNN()
        self.loss_fn = FocalLoss(alpha=0.5, gamma=3, reduction='mean', pos_weight=torch.tensor([pos_weight]))
        self.distill = nn.MSELoss()
        self.train_acc = Accuracy(task='binary'); self.val_acc = Accuracy(task='binary')
        self.val_f1 = F1Score(task='binary'); self.val_auc = AUROC(task='binary')
        self.val_probs, self.val_labels = [], []
        self.hparams.best_val_threshold = 0.5
        self.hparams.val_f1_at_optimal_thresh = 0.0
    def forward(self, x, g): return self.model(x, g)
    def _mixup(self, x, g, y):
        lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA) if MIXUP_ALPHA>0 else 1.
        idx = torch.randperm(x.size(0), device=x.device)
        x2 = lam*x + (1-lam)*x[idx]; g2 = g if g.numel()==0 else lam*g + (1-lam)*g[idx]
        return x2, g2, y, y[idx], lam
    def _mixup_loss(self, crit, pred, ya, yb, lam):
        crit_none = FocalLoss(alpha=crit.alpha, gamma=crit.gamma, reduction='none', pos_weight=crit.pos_weight)
        return (lam*crit_none(pred, ya).mean() + (1-lam)*crit_none(pred, yb).mean())
    def _step(self, batch, stage):
        x,g,y = batch
        x,g,y = x.to(self.device), g.to(self.device), y.to(self.device)
        if USE_SEQ2SEQ_PREDICTION:
            target_len = EEG2REP_REDUCED_TIME_STEPS
            y2 = F.adaptive_avg_pool1d(y.unsqueeze(0).unsqueeze(0).float(), target_len).squeeze(0).squeeze(0)
            ybin = y2.round().long()
        else:
            y2 = y.float().unsqueeze(-1); ybin = y2.round().long()
        if stage=='train' and USE_MIXUP:
            x2,g2,ya,yb,lam = self._mixup(x,g,y2)
            logits,_ = self(x2,g2)
            bce = self._mixup_loss(self.loss_fn, logits, ya, yb, lam)
        else:
            logits,_ = self(x,g); bce = self.loss_fn(logits, y2)
        loss = bce
        probs = torch.sigmoid(logits)
        if USE_SEQ2SEQ_PREDICTION:
            pf = probs.view(-1); yf = ybin.view(-1)
        else:
            pf = probs.squeeze(1); yf = ybin.squeeze(1)
        getattr(self, f"{stage}_acc").update(pf, yf)
        if stage=='val':
            self.val_f1.update(pf, yf); self.val_auc.update(pf, yf)
            self.val_probs.append(pf.detach().cpu()); self.val_labels.append(yf.detach().cpu())
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=self.hparams.get('batch_size', BATCH_SIZE))
        return loss
    def training_step(self, b, i): return self._step(b,'train')
    def validation_step(self, b, i): return self._step(b,'val')
    def on_validation_epoch_end(self):
        if not self.val_probs: return
        probs = torch.cat(self.val_probs).numpy(); labels = torch.cat(self.val_labels).numpy()
        self.val_probs.clear(); self.val_labels.clear()
        best_f1m, best_t = 0, 0.5
        if len(np.unique(labels))>=2:
            for t in np.linspace(0.01,0.99,100):
                preds = (probs>=t).astype(int)
                f1p = f1_score(labels, preds, pos_label=1, average='binary', zero_division=0)
                f1n = f1_score(labels, preds, pos_label=0, average='binary', zero_division=0)
                f1m = 0.5*(f1p+f1n)
                if f1m>best_f1m: best_f1m, best_t = f1m, t
        self.hparams.best_val_threshold = float(best_t)
        self.hparams.val_f1_at_optimal_thresh = float(best_f1m)
        self.log_dict({
            "val_acc": self.val_acc.compute(),
            "val_f1_at_optimal_thresh": best_f1m,
            "val_auc": self.val_auc.compute(),
            "val_optimal_threshold": best_t
        }, prog_bar=True)
        self.val_acc.reset(); self.val_f1.reset(); self.val_auc.reset()
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
        sch = CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

## Data & Training
# Train/Val splits per your script
train_keys = [( "0", f"{i}", "Sherlock2", "1" ) for i in range(1,13) if i != 2] \               + [( "0", f"{i}", "Sherlock4", "1" ) for i in range(1,13) if i != 8] \               + [( "0", f"{i}", "Sherlock5", "1" ) for i in range(1,13)] \               + [( "0", f"{i}", "Sherlock6", "1" ) for i in range(2,13)]
val_keys = [("0","11","Sherlock1","2")]

def calculate_pos_weight(loader):
    pos=0; neg=0
    for _,_,y in loader:
        pos += torch.sum(y==1); neg += torch.sum(y==0)
    return (neg/max(pos, torch.tensor(1))).float()

class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, is_train=False, **aug):
        super().__init__()
        self.ds = dataset; self.is_train=is_train; self.aug=aug
        self.idx = list(range(len(dataset)))
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        x,y = self.ds[self.idx[i]]
        x = x.cpu().float(); y=y.cpu().float()
        cur = x.shape[1]
        if cur<SEGMENT_TIME_LEN_SAMPLES:
            x = F.pad(x, (0, SEGMENT_TIME_LEN_SAMPLES-cur)); y = F.pad(y, (0, SEGMENT_TIME_LEN_SAMPLES-cur))
        elif cur>SEGMENT_TIME_LEN_SAMPLES:
            x = x[:, :SEGMENT_TIME_LEN_SAMPLES]; y = y[:SEGMENT_TIME_LEN_SAMPLES]
        # GNN features (downsampled + PCA placeholder)
        gnn_feats = []
        x_masked = x[SENSORS_SPEECH_MASK].float()
        for ch in range(NUM_RAW_CHANNELS_CNN_STREAM):
            dsig = (x_masked[ch,:]).float()
            dsig = downsample_raw_data(dsig, RAW_DOWNSAMPLE_LEN)
            pad = F.pad(dsig, (0, GNN_INPUT_NODE_FEATURE_COMMON_DIM-RAW_DOWNSAMPLE_LEN))
            gnn_feats.append(pad)
        # clusters (zeros placeholder here; build real in utils notebook or reuse built globals)
        for _ in range(NUM_GNN_SPATIAL_CLUSTER_NODES):
            gnn_feats.append(torch.zeros(GNN_INPUT_NODE_FEATURE_COMMON_DIM))
        gnn_feats = torch.stack(gnn_feats) if gnn_feats else torch.empty(0)
        return x, gnn_feats, y

train_raw = LibriBrainSpeech(data_path=f"{BASE_PATH}/data/", include_run_keys=train_keys, tmin=0.0, tmax=4.0, preload_files=False, standardize=True, oversample_silence_jitter=70)
val_raw   = LibriBrainSpeech(data_path=f"{BASE_PATH}/data/", include_run_keys=val_keys,   tmin=0.0, tmax=4.0, preload_files=False, standardize=True, oversample_silence_jitter=0)
train_ds  = FilteredDataset(train_raw, is_train=True)
val_ds    = FilteredDataset(val_raw,   is_train=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

pos_w = calculate_pos_weight(train_loader)
model = SpeechClassifier(pos_weight=pos_w, batch_size=BATCH_SIZE)
logger = CSVLogger(save_dir=f"{BASE_PATH}/lightning_logs", name="", version="fine_tuned_eeg2rep_hybrid_speech")
ckpt = ModelCheckpoint(dirpath=f"{BASE_PATH}/models", filename="best_fine_tuned_eeg2rep_hybrid_speech_{epoch:02d}-{val_f1_at_optimal_thresh:.3f}", monitor="val_f1_at_optimal_thresh", mode="max", save_top_k=1)
trainer = L.Trainer(devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu", max_epochs=NUM_EPOCHS,
                    logger=logger, callbacks=[EarlyStopping("val_f1_at_optimal_thresh", patience=10, mode="max"), ckpt, TQDMProgressBar(refresh_rate=1), StochasticWeightAveraging(swa_lrs=1e-3)])
print("Ready to train. (Skip execution here if you just wanted the notebook files.)")
