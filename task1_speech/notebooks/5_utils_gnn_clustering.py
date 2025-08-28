# 5 — GNN Utilities: Sensor Positions, K-Means Clustering, and Static Graph
# Builds spatial clusters and a static graph used by the GNN branch.

import os
import json
from typing import List, Sequence, Optional

import numpy as np
import torch
import torch_geometric.utils
from sklearn.cluster import KMeans


def load_sensor_positions(base_path: str, filename: str = "sensor_xyz.json") -> np.ndarray:
    """
    Load MEG sensor 3D positions from JSON.
    Expected format: a list/array of shape (306, 3).
    """
    p = os.path.join(base_path, filename)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Missing sensor XYZ file at {p}. "
            "Download it (per competition docs) and place it in base_path."
        )
    with open(p, "r") as fp:
        pos = np.array(json.load(fp), dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"sensor_xyz has invalid shape: {pos.shape}. Expected (N, 3).")
    return pos


def perform_spatial_clustering(
    sensor_xyz: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
) -> List[List[int]]:
    """
    K-means clustering over sensor (x,y,z) to get spatial groups.
    Returns a list of clusters; each cluster is a list of channel indices.
    """
    if sensor_xyz.ndim != 2 or sensor_xyz.shape[1] != 3:
        raise ValueError(f"sensor_xyz must be (N, 3). Got {sensor_xyz.shape}.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(sensor_xyz)
    clusters: List[List[int]] = [[] for _ in range(n_clusters)]
    for idx, c in enumerate(labels):
        clusters[c].append(int(idx))
    return clusters


def build_static_adjacency_matrix(
    num_raw_masked_nodes: int,
    num_spatial_cluster_nodes: int,
    spatial_assignments: Sequence[Sequence[int]],
    sensors_speech_mask: Sequence[int],
) -> torch.Tensor:
    """
    Build a static, undirected graph with:
      • fully-connected raw nodes (0..num_raw_masked_nodes-1)
      • fully-connected cluster nodes (offset by num_raw_masked_nodes)
      • cross-edges from each raw node to its cluster node (membership by channel id)
    Returns edge_index with shape (2, E) suitable for torch_geometric.
    """
    edges = []

    # 1) fully-connect raw nodes
    for i in range(num_raw_masked_nodes):
        for j in range(i + 1, num_raw_masked_nodes):
            edges += [[i, j], [j, i]]

    # 2) fully-connect cluster nodes
    offset = num_raw_masked_nodes
    for i in range(num_spatial_cluster_nodes):
        for j in range(i + 1, num_spatial_cluster_nodes):
            a = offset + i
            b = offset + j
            edges += [[a, b], [b, a]]

    # 3) raw-to-cluster edges by membership
    # sensors_speech_mask contains the original channel ids for each raw node index
    for raw_idx, ch in enumerate(sensors_speech_mask):
        for cid, members in enumerate(spatial_assignments):
            if ch in members:
                a = raw_idx
                b = offset + cid
                edges += [[a, b], [b, a]]
                break

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    ei, _ = torch_geometric.utils.remove_self_loops(ei)
    ei = torch_geometric.utils.to_undirected(ei)
    ei = torch_geometric.utils.sort_edge_index(ei)
    return ei

