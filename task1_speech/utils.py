# libribrain/utils.py
import os
import json
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch_geometric import utils as pyg_utils

# --- Global GNN Placeholders ---
SENSOR_XYZ_POSITIONS = None
SPATIAL_CLUSTER_ASSIGNMENTS = None

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss
        return f_loss.mean()

def calculate_pos_weight(loader):
    pos_count, total_count = 0, 0
    for _, _, y in tqdm(loader, desc="Calculating POS_WEIGHT"):
        pos_count += torch.sum(y == 1)
        total_count += y.numel()
    neg_count = total_count - pos_count
    return neg_count / pos_count if pos_count > 0 else torch.tensor(1.0)

def downsample_raw_data(tensor, target_len):
    original_len = tensor.size(0)
    if original_len == target_len: return tensor
    if original_len < target_len: return F.pad(tensor, (0, target_len - original_len))
    
    chunk_size = original_len // target_len
    return tensor[:(chunk_size * target_len)].view(target_len, chunk_size).mean(dim=1)

def _load_sensor_positions(cfg):
    global SENSOR_XYZ_POSITIONS
    path = os.path.join(cfg.BASE_PATH, "sensor_xyz.json")
    if not os.path.exists(path):
        print("Downloading sensor_xyz.json...")
        os.makedirs(cfg.BASE_PATH, exist_ok=True)
        resp = requests.get("https://neural-processing-lab.github.io/2025-libribrain-competition/sensor_xyz.json")
        with open(path, "wb") as f: f.write(resp.content)
    with open(path, "r") as f: SENSOR_XYZ_POSITIONS = np.array(json.load(f))

def _perform_spatial_clustering(cfg):
    global SPATIAL_CLUSTER_ASSIGNMENTS
    kmeans = KMeans(n_clusters=cfg.NUM_GNN_SPATIAL_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(SENSOR_XYZ_POSITIONS)
    SPATIAL_CLUSTER_ASSIGNMENTS = [[] for _ in range(cfg.NUM_GNN_SPATIAL_CLUSTERS)]
    for i, cluster_id in enumerate(labels):
        SPATIAL_CLUSTER_ASSIGNMENTS[cluster_id].append(i)

def build_static_adjacency_matrix(cfg):
    edge_list = []
    # Connect raw-masked nodes to their spatial cluster
    for raw_idx, ch_id in enumerate(cfg.SENSORS_SPEECH_MASK):
        for cluster_id, channels in enumerate(SPATIAL_CLUSTER_ASSIGNMENTS):
            if ch_id in channels:
                edge_list.append([raw_idx, cfg.NUM_RAW_CHANNELS_CNN_STREAM + cluster_id])
                break
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return pyg_utils.to_undirected(edge_index)

def setup_gnn_environment(cfg):
    print("Setting up GNN environment...")
    _load_sensor_positions(cfg)
    _perform_spatial_clustering(cfg)
    return build_static_adjacency_matrix(cfg)
