# utils.py
import os
import json
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config

# Global variables to be populated dynamically
SPATIAL_CLUSTER_ASSIGNMENTS = None
SENSOR_XYZ_POSITIONS = None
STATIC_GNN_EDGE_INDEX = None

def calculate_pos_weight(train_loader):
    print("Calculating positive weight for loss function...")
    pos_count, neg_count = 0, 0
    for _, _, y in tqdm(train_loader, desc="Calculating POS_WEIGHT"):
        pos_count += torch.sum(y == 1)
        neg_count += torch.sum(y == 0)

    if pos_count == 0:
        return torch.tensor(1.0)
    
    pos_weight_value = neg_count / pos_count
    print(f"Negative samples: {neg_count}, Positive samples: {pos_count}")
    print(f"Calculated POS_WEIGHT: {pos_weight_value:.4f}")
    return pos_weight_value

def downsample_raw_data(channel_data_tensor, target_len):
    channel_data_tensor = channel_data_tensor.float()
    original_len = channel_data_tensor.size(0)
    if original_len < target_len:
        padding = target_len - original_len
        return F.pad(channel_data_tensor, (0, padding), 'constant', 0.0)
    chunk_size = original_len // target_len
    trimmed_data = channel_data_tensor[:(chunk_size * target_len)]
    return trimmed_data.view(target_len, chunk_size).mean(dim=1)

def load_sensor_positions():
    global SENSOR_XYZ_POSITIONS
    p = os.path.join(config.BASE_PATH, "sensor_xyz.json")
    if not os.path.exists(p):
        print("Downloading sensor_xyz.json...")
        os.makedirs(config.BASE_PATH, exist_ok=True)
        with open(p, "wb") as f:
            f.write(requests.get("https://neural-processing-lab.github.io/2025-libribrain-competition/sensor_xyz.json").content)
    with open(p, "r") as fp:
        SENSOR_XYZ_POSITIONS = np.array(json.load(fp))

def perform_spatial_clustering():
    global SPATIAL_CLUSTER_ASSIGNMENTS
    if SENSOR_XYZ_POSITIONS is None: load_sensor_positions()
    print(f"\n--- Starting Spatial K-Means Clustering (K={config.NUM_GNN_SPATIAL_CLUSTERS}) ---")
    kmeans = KMeans(n_clusters=config.NUM_GNN_SPATIAL_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(SENSOR_XYZ_POSITIONS)
    SPATIAL_CLUSTER_ASSIGNMENTS = [[] for _ in range(config.NUM_GNN_SPATIAL_CLUSTERS)]
    for i, cluster_id in enumerate(cluster_labels):
        SPATIAL_CLUSTER_ASSIGNMENTS[cluster_id].append(i)
    print("--- Spatial Clustering Complete ---")

def build_static_adjacency_matrix():
    global STATIC_GNN_EDGE_INDEX
    edge_list = []
    # Fully connect raw-masked nodes and spatial cluster nodes among themselves
    for i in range(config.NUM_RAW_CHANNELS_CNN_STREAM):
        for j in range(i + 1, config.NUM_RAW_CHANNELS_CNN_STREAM):
            edge_list.append([i, j])
    for i in range(config.NUM_GNN_SPATIAL_CLUSTER_NODES):
        for j in range(i + 1, config.NUM_GNN_SPATIAL_CLUSTER_NODES):
            edge_list.append([config.NUM_RAW_CHANNELS_CNN_STREAM + i, config.NUM_RAW_CHANNELS_CNN_STREAM + j])
            
    # Connect raw-masked nodes to the spatial cluster they belong to
    for raw_idx, original_ch_id in enumerate(config.SENSORS_SPEECH_MASK):
        for cluster_id, channels in enumerate(SPATIAL_CLUSTER_ASSIGNMENTS):
            if original_ch_id in channels:
                edge_list.append([raw_idx, config.NUM_RAW_CHANNELS_CNN_STREAM + cluster_id])
                break
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    STATIC_GNN_EDGE_INDEX, _ = torch_geometric.utils.remove_self_loops(edge_index)

def get_teacher_signal_for_batch(labels, device):
    kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=device).view(1, 1, -1)
    padding = (kernel.shape[2] - 1) // 2
    if labels.dim() == 1:
        labels = labels.unsqueeze(1).repeat(1, config.SEGMENT_TIME_LEN_SAMPLES)
    smoothed_labels = F.conv1d(labels.unsqueeze(1).float(), kernel, padding=padding)
    return smoothed_labels.squeeze(1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * ((1 - pt) ** self.gamma) * BCE_loss
        return F_loss.mean()
