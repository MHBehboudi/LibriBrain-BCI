# If you need to install, uncomment for your own environment:
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install lightning torchmetrics torch-geometric
# !pip install scikit-learn pandas numpy tqdm
# !pip install pnpl

import os, math, random, argparse, json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from pnpl.datasets import LibriBrainSpeech, LibriBrainCompetitionHoldout

from torchmetrics import Accuracy, F1Score, AUROC

# GNN
from torch_geometric.nn import GCNConv, GATConv
import torch_geometric.utils

# Sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# Lightning callbacks/loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, StochasticWeightAveraging
from lightning.pytorch.loggers import CSVLogger

print('PyTorch:', torch.__version__)
print('Lightning:', L.__version__)

## Paths & Global Config
BASE_PATH = "../scratch/libribrain_data"
SUBMISSION_PATH = "libribrain_data/submissions"
PRETRAIN_CHECKPOINT_PATH = os.path.join(SUBMISSION_PATH, "eeg2rep_pretrain.ckpt")

os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(SUBMISSION_PATH, exist_ok=True)

# Model & Training Parameters 
NUM_EPOCHS = 5
PRETRAIN_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
PRETRAIN_LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.3
LAMBDA_SPARSE = 1e-5
LSTM_LAYERS = 2
BI_DIRECTIONAL = True
POS_WEIGHT = 3.0  

CONV_DIM = 256
ATTN_DIM = 128
TRANSFORMER_NHEAD = 4
TRANSFORMER_DIM_FEEDFORWARD = 512

# Feature & Architecture Flags 
USE_FILTER_BANK_INPUT = False
USE_SEQ2SEQ_PREDICTION = True
USE_KNOWLEDGE_DISTILLATION = True

# GNN & Residual & Fusion Attention Flags
USE_GNN = True
USE_GAT = True
GNN_OUTPUT_DIM = 64
RAW_DOWNSAMPLE_LEN = 10
NUM_GNN_LAYERS = 3

USE_CNN_RESIDUAL = True
USE_GNN_RESIDUAL = True
USE_FUSION_ATTENTION = True
FUSION_ATTN_HEADS = 4
FUSION_ATTN_DIM_PER_HEAD = 16

# K-Means / PCA for GNN 
NUM_GNN_SPATIAL_CLUSTERS = 50
NUM_GNN_SPATIAL_CLUSTER_NODES = NUM_GNN_SPATIAL_CLUSTERS
NUM_PCA_COMPONENTS_PER_CLUSTER = 3
USE_PCA_VARIANCE_AS_FEATURE = False

# EEG2Rep Params 
EEG2REP_EMBEDDING_DIM = 256
EEG2REP_REDUCED_TIME_STEPS = 50
EEG2REP_TRANSFORMER_HEADS = 8
EEG2REP_TRANSFORMER_LAYERS = 8
EEG2REP_MASK_RATIO = 0.6
EEG2REP_PREDICTOR_DEPTH = 2

# Distillation / Augs 
DISTILLATION_ALPHA = 0.7
USE_MIXUP = True
MIXUP_ALPHA = 0.4
APPLY_GAUSSIAN_NOISE = True
NOISE_STD = 0.015
ENABLE_SWA = True

# Augs
AUG_RANDOM_CROP_MIN_RATIO = 0.8
AUG_RANDOM_CROP_MAX_RATIO = 1.0
AUG_TIME_WARP_MAX_STRETCH = 0.05
AUG_CHANNEL_MASK_RATIO = 0.2

# Debug
DEBUG_PREDICTION_SAMPLES = None

# Sensors
SENSORS_SPEECH_MASK = [
    18, 20, 22, 23, 45, 120, 138, 140, 142, 143, 145,
    146, 147, 149, 175, 176, 177, 179, 180, 198, 271, 272, 275
]
NUM_RAW_CHANNELS_CNN_STREAM = len(SENSORS_SPEECH_MASK)
NUM_TOTAL_MEG_CHANNELS = 306

# Timebase
SAMPLING_RATE = 250
SEGMENT_TIME_LEN_SAMPLES = 1000
SEGMENT_TIME_LEN_SECONDS = SEGMENT_TIME_LEN_SAMPLES / SAMPLING_RATE

# Globals (populated later)
SPATIAL_CLUSTER_ASSIGNMENTS = None
SENSOR_XYZ_POSITIONS = None
STATIC_GNN_EDGE_INDEX = None

# CNN-Attn Transformer depth
CNN_ATTN_TRANSFORMER_LAYERS = 8

# Derived
if USE_PCA_VARIANCE_AS_FEATURE:
    PCA_CLUSTER_FEATURE_LEN = NUM_PCA_COMPONENTS_PER_CLUSTER + 1
else:
    PCA_CLUSTER_FEATURE_LEN = NUM_PCA_COMPONENTS_PER_CLUSTER

GNN_INPUT_NODE_FEATURE_COMMON_DIM = max(RAW_DOWNSAMPLE_LEN, PCA_CLUSTER_FEATURE_LEN)
if USE_GNN:
    NUM_GNN_NODES_TOTAL = NUM_RAW_CHANNELS_CNN_STREAM + NUM_GNN_SPATIAL_CLUSTER_NODES
else:
    NUM_GNN_NODES_TOTAL = 0

print('NUM_GNN_NODES_TOTAL:', NUM_GNN_NODES_TOTAL)

## Reproducibility
L.seed_everything(42)
torch.set_float32_matmul_precision("high")
