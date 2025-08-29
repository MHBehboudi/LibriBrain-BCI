# data.py
import torch
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pnpl.datasets import LibriBrainSpeech, LibriBrainCompetitionHoldout

import config
import utils

class FilteredDataset(Dataset):
    def __init__(self, base_dataset, is_train=False):
        self.base_dataset = base_dataset
        self.is_train = is_train

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sensors_raw, labels_seq = self.base_dataset[idx]
        
        # Ensure correct length
        if sensors_raw.shape[1] < config.SEGMENT_TIME_LEN_SAMPLES:
            pad_len = config.SEGMENT_TIME_LEN_SAMPLES - sensors_raw.shape[1]
            sensors_raw = F.pad(sensors_raw, (0, pad_len), 'constant', 0.0)
            labels_seq = F.pad(labels_seq, (0, pad_len), 'constant', 0.0)
        else:
            sensors_raw = sensors_raw[:, :config.SEGMENT_TIME_LEN_SAMPLES]
            labels_seq = labels_seq[:config.SEGMENT_TIME_LEN_SAMPLES]

        # Augmentations for training data
        if self.is_train:
            if config.APPLY_GAUSSIAN_NOISE:
                sensors_raw += torch.randn_like(sensors_raw) * config.NOISE_STD
            if config.AUG_CHANNEL_MASK_RATIO > 0:
                num_mask = int(sensors_raw.shape[0] * config.AUG_CHANNEL_MASK_RATIO)
                masked_channels = random.sample(range(sensors_raw.shape[0]), num_mask)
                sensors_raw[masked_channels, :] = 0.0

        gnn_features = torch.empty(0)
        if config.USE_GNN:
            node_features = []
            # Raw sensor features
            sensors_masked_for_gnn = sensors_raw[config.SENSORS_SPEECH_MASK]
            for i in range(config.NUM_RAW_CHANNELS_CNN_STREAM):
                downsampled = utils.downsample_raw_data(sensors_masked_for_gnn[i], config.RAW_DOWNSAMPLE_LEN)
                node_features.append(downsampled)
            # Spatial cluster features
            for cluster_channels in utils.SPATIAL_CLUSTER_ASSIGNMENTS:
                if len(cluster_channels) > 1:
                    cluster_data = sensors_raw[cluster_channels].numpy().T
                    scaled_data = StandardScaler().fit_transform(cluster_data).T
                    _, S, _ = torch.pca_lowrank(torch.from_numpy(scaled_data).float(), q=config.NUM_PCA_COMPONENTS_PER_CLUSTER)
                    node_features.append(S)
                else:
                    node_features.append(torch.zeros(config.NUM_PCA_COMPONENTS_PER_CLUSTER))
            gnn_features = torch.stack(node_features)

        return sensors_raw.float(), gnn_features.float(), labels_seq.float()

def make_loaders(batch_size):
    train_keys = [( "0", f"{i}", "Sherlock2", "1" ) for i in range(1,13) if i != 2] + [( "0", f"{i}", "Sherlock4", "1" ) for i in range(1,13) if i != 8] + [( "0", f"{i}", "Sherlock5", "1" ) for i in range(1,13)] +[( "0", f"{i}", "Sherlock6", "1" ) for i in range(2,13)]
    val_keys = [("0","11","Sherlock1","2")]
    
    train_raw = LibriBrainSpeech(data_path=f"{config.BASE_PATH}/data/", include_run_keys=train_keys, tmin=0.0, tmax=4.0, standardize=True, oversample_silence_jitter=70)
    val_raw = LibriBrainSpeech(data_path=f"{config.BASE_PATH}/data/", include_run_keys=val_keys, tmin=0.0, tmax=4.0, standardize=True)
    
    train_ds = FilteredDataset(train_raw, is_train=True)
    val_ds = FilteredDataset(val_raw, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return {"train": train_loader, "val": val_loader}
