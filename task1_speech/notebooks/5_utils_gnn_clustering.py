# 5 — GNN Utilities: Sensor Positions, K-Means Clustering, and Static Graph
## Builds spatial clusters and a static graph used by the GNN branch.

import os, json, torch, numpy as np
import torch_geometric.utils
from sklearn.cluster import KMeans

def load_sensor_positions():
    global SENSOR_XYZ_POSITIONS
    p = os.path.join(BASE_PATH, "sensor_xyz.json")
    if not os.path.exists(p):
        print("Place 'sensor_xyz.json' at:", p, "(download externally).")
    with open(p, "r") as fp:
        pos = np.array(json.load(fp))
    return pos

def perform_spatial_clustering(sensor_xyz, k=NUM_GNN_SPATIAL_CLUSTERS):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(sensor_xyz)
    clusters = [[] for _ in range(k)]
    for i,c in enumerate(labels): clusters[c].append(i)
    return clusters

def build_static_adjacency_matrix(num_raw_masked_nodes, num_spatial_cluster_nodes, spatial_assignments):
    total_nodes = num_raw_masked_nodes + num_spatial_cluster_nodes
    edges = []
    # fully connect raw nodes
    for i in range(num_raw_masked_nodes):
        for j in range(i+1, num_raw_masked_nodes):
            edges += [[i,j],[j,i]]
    # fully connect cluster nodes
    for i in range(num_spatial_cluster_nodes):
        for j in range(i+1, num_spatial_cluster_nodes):
            a = num_raw_masked_nodes+i; b=num_raw_masked_nodes+j
            edges += [[a,b],[b,a]]
    # cross connections by membership
    for raw_idx, ch in enumerate(SENSORS_SPEECH_MASK):
        for cid, members in enumerate(spatial_assignments):
            if ch in members:
                a = raw_idx; b = num_raw_masked_nodes+cid
                edges += [[a,b],[b,a]]; break
    if not edges:
        return torch.empty((2,0), dtype=torch.long)
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    ei, _ = torch_geometric.utils.remove_self_loops(ei)
    ei = torch_geometric.utils.to_undirected(ei)
    ei = torch_geometric.utils.sort_edge_index(ei)
    return ei

# Example flow:
# xyz = load_sensor_positions()
# clusters = perform_spatial_clustering(xyz, k=NUM_GNN_SPATIAL_CLUSTERS)
# STATIC_GNN_EDGE_INDEX = build_static_adjacency_matrix(NUM_RAW_CHANNELS_CNN_STREAM, NUM_GNN_SPATIAL_CLUSTER_NODES, clusters)
# print(STATIC_GNN_EDGE_INDEX.shape)
