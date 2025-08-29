# run.py
import argparse
import os
import torch
from libribrain import config, utils
from libribrain.data_manager import DataManager
from libribrain.pipeline import Trainer, Ensembler

def main():
    parser = argparse.ArgumentParser(description="LibriBrain Modular Project Runner")
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--speech_ckpt", type=str, default=f"{config.MODELS_PATH}/best_speech_focused.ckpt")
    parser.add_argument("--silence_ckpt", type=str, default=f"{config.MODELS_PATH}/best_silence_focused.ckpt")
    args = parser.parse_args()

    # --- Setup ---
    for path in [config.MODELS_PATH, config.SUBMISSIONS_PATH, config.LOGS_PATH]:
        os.makedirs(path, exist_ok=True)
    
    gnn_edge_index = utils.setup_gnn_environment(config) if config.USE_GNN else None
    data_manager = DataManager()
    
    # --- Command Execution ---
    if args.command == "train":
        # Get data loaders and dynamically calculated pos_weight
        _, _ = data_manager.get_train_val_loaders()
        pos_weight = data_manager.pos_weight
        inv_pos_weight = 1.0 / pos_weight if pos_weight > 0 else 1.0
        
        # Train speech-focused model
        Trainer("speech_focused", pos_weight, gnn_edge_index).run(data_manager)
        
        # Train silence-focused model
        Trainer("silence_focused", inv_pos_weight, gnn_edge_index).run(data_manager)
        
    elif args.command == "predict":
        if not (os.path.exists(args.speech_ckpt) and os.path.exists(args.silence_ckpt)):
            raise FileNotFoundError("Checkpoints not found. Please run 'train' first or provide correct paths.")
        
        Ensembler(gnn_edge_index).predict(args.speech_ckpt, args.silence_ckpt, data_manager)

if __name__ == "__main__":
    main()
