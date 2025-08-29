# libribrain/config.py
import os

# --- Core Paths ---
BASE_PATH = os.path.join(os.path.expanduser("~"), "scratch", "libribrain_data")
MODELS_PATH = os.path.join(BASE_PATH, "models")
SUBMISSIONS_PATH = os.path.join(BASE_PATH, "submissions")
LOGS_PATH = os.path.join(BASE_PATH, "logs")

# --- Model & Training Parameters ---
NUM_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.4
LAMBDA_SPARSE = 1e-5
LSTM_LAYERS = 2
BI_DIRECTIONAL = True

# --- Main Temporal Encoder Parameters ---
TEMPORAL_EMBEDDING_DIM = 256
TEMPORAL_TRANSFORMER_HEADS = 8
TEMPORAL_TRANSFORMER_LAYERS = 6
TEMPORAL_REDUCED_TIME_STEPS = 50

# --- CNN/Attention Stream Parameters ---
CONV_DIM_CNN_STREAM = 256
ATTN_DIM_CNN_STREAM = 128
TRANSFORMER_NHEAD_CNN_STREAM = 4
TRANSFORMER_DIM_FF_CNN_STREAM = 512
CNN_ATTN_TRANSFORMER_LAYERS = 6

# --- GNN & Residual & Fusion Attention Flags ---
USE_GNN = True
USE_GAT = True # Set to False to use GCN
GNN_OUTPUT_DIM = 64
RAW_DOWNSAMPLE_LEN = 10
NUM_GNN_LAYERS = 3
USE_GNN_RESIDUAL = True

# --- GNN Clustering Parameters ---
NUM_GNN_SPATIAL_CLUSTERS = 50
NUM_PCA_COMPONENTS_PER_CLUSTER = 3

# --- Other Parameters ---
USE_KNOWLEDGE_DISTILLATION = True
DISTILLATION_ALPHA = 0.7

# --- FIXED SENSOR SET & DERIVED PARAMS ---
SENSORS_SPEECH_MASK = [
    18, 20, 22, 23, 45, 120, 138, 140, 142, 143, 145,
    146, 147, 149, 175, 176, 177, 179, 180, 198, 271, 272, 275
]
NUM_RAW_CHANNELS_CNN_STREAM = len(SENSORS_SPEECH_MASK)
NUM_GNN_SPATIAL_CLUSTER_NODES = NUM_GNN_SPATIAL_CLUSTERS
NUM_GNN_NODES_TOTAL = NUM_RAW_CHANNELS_CNN_STREAM + NUM_GNN_SPATIAL_CLUSTER_NODES if USE_GNN else 0
SEGMENT_TIME_LEN_SAMPLES = 1000

# --- DATA KEYS ---
TRAIN_KEYS = [( "0", f"{i}", "Sherlock2", "1" ) for i in range(1,13) if i != 2] + [( "0", f"{i}", "Sherlock4", "1" ) for i in range(1,13) if i != 8] + [( "0", f"{i}", "Sherlock5", "1" ) for i in range(1,13)] +[( "0", f"{i}", "Sherlock6", "1" ) for i in range(2,13)]
VAL_KEYS = [("0","11","Sherlock1","2")]
