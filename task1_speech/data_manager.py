# libribrain/data_manager.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pnpl.datasets import LibriBrainSpeech, LibriBrainCompetitionHoldout
from sklearn.preprocessing import StandardScaler
from . import config as cfg
from . import utils

class SpeechDataset(Dataset):
    def __init__(self, base_dataset, is_train=False):
        self.base = base_dataset
        self.is_train = is_train

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sensors, labels = self.base[idx]
        if sensors.shape[1] < cfg.SEGMENT_TIME_LEN_SAMPLES:
            pad = cfg.SEGMENT_TIME_LEN_SAMPLES - sensors.shape[1]
            sensors = F.pad(sensors, (0, pad))
            labels = F.pad(labels, (0, pad))
        else:
            sensors = sensors[:, :cfg.SEGMENT_TIME_LEN_SAMPLES]
            labels = labels[:cfg.SEGMENT_TIME_LEN_SAMPLES]
        
        gnn_features = torch.empty(0)
        if cfg.USE_GNN:
            nodes = []
            masked_sensors = sensors[cfg.SENSORS_SPEECH_MASK]
            for i in range(cfg.NUM_RAW_CHANNELS_CNN_STREAM):
                nodes.append(utils.downsample_raw_data(masked_sensors[i], cfg.RAW_DOWNSAMPLE_LEN))
            for cluster in utils.SPATIAL_CLUSTER_ASSIGNMENTS:
                if len(cluster) > 1:
                    data = StandardScaler().fit_transform(sensors[cluster].numpy().T).T
                    _, S, _ = torch.pca_lowrank(torch.from_numpy(data).float(), q=cfg.NUM_PCA_COMPONENTS_PER_CLUSTER)
                    nodes.append(S)
                else:
                    nodes.append(torch.zeros(cfg.NUM_PCA_COMPONENTS_PER_CLUSTER))
            gnn_features = torch.stack([F.pad(n, (0, cfg.RAW_DOWNSAMPLE_LEN - n.shape[0])) for n in nodes])
            
        return sensors.float(), gnn_features.float(), labels.float()

class DataManager:
    def __init__(self):
        self.pos_weight = None

    def get_train_val_loaders(self):
        train_raw = LibriBrainSpeech(data_path=f"{cfg.BASE_PATH}/data/", include_run_keys=cfg.TRAIN_KEYS, tmin=0.0, tmax=4.0, standardize=True, oversample_silence_jitter=70)
        val_raw = LibriBrainSpeech(data_path=f"{cfg.BASE_PATH}/data/", include_run_keys=cfg.VAL_KEYS, tmin=0.0, tmax=4.0, standardize=True)
        train_ds = SpeechDataset(train_raw, is_train=True)
        val_ds = SpeechDataset(val_raw)
        
        temp_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, num_workers=2)
        self.pos_weight = utils.calculate_pos_weight(temp_loader)
        
        return (DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4),
                DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, num_workers=4))

    def get_holdout_loader(self):
        # Implement SlidingWindowDataset if needed, or simplify for now
        holdout_raw = LibriBrainCompetitionHoldout(data_path=f"{cfg.BASE_PATH}/data/", task="speech", standardize=True)
        holdout_ds = SpeechDataset(holdout_raw) # Simplified for this example
        return DataLoader(holdout_ds, batch_size=cfg.BATCH_SIZE, num_workers=4)
