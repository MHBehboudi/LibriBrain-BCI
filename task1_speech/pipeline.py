# libribrain/pipeline.py
import os
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import pandas as pd
import numpy as np
from tqdm import tqdm

from . import config as cfg, model, utils
from .data_manager import DataManager

class Trainer:
    def __init__(self, model_name, pos_weight, gnn_edge_index):
        self.model_name = model_name
        self.pos_weight = pos_weight
        self.gnn_edge_index = gnn_edge_index

    def run(self, data_manager):
        train_loader, val_loader = data_manager.get_train_val_loaders()
        
        hparams = {k: v for k, v in vars(cfg).items() if not k.startswith('__')}
        hparams['pos_weight'] = self.pos_weight
        
        classifier = model.SpeechClassifier(hparams, self.gnn_edge_index)
        
        logger = CSVLogger(save_dir=cfg.LOGS_PATH, name=self.model_name)
        ckpt_cb = ModelCheckpoint(dirpath=cfg.MODELS_PATH, filename=f"best_{self.model_name}", monitor="val_f1_optimal", mode="max")
        
        trainer = L.Trainer(max_epochs=cfg.NUM_EPOCHS, logger=logger, callbacks=[EarlyStopping("val_f1_optimal", patience=5, mode="max"), ckpt_cb], accelerator="auto")
        
        print(f"--- 🚀 Starting training: {self.model_name} ---")
        trainer.fit(classifier, train_loader, val_loader)
        print(f"--- ✅ Training complete: {self.model_name}. Model saved to {ckpt_cb.best_model_path} ---")

class Ensembler:
    def __init__(self, gnn_edge_index):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gnn_edge_index = gnn_edge_index.to(self.device) if gnn_edge_index is not None else None

    def predict(self, speech_ckpt, silence_ckpt, data_manager):
        holdout_loader = data_manager.get_holdout_loader()
        all_probs = []

        for name, ckpt_path in [("Speech", speech_ckpt), ("Silence", silence_ckpt)]:
            classifier = model.SpeechClassifier.load_from_checkpoint(ckpt_path, gnn_edge_index=self.gnn_edge_index).to(self.device)
            classifier.eval()
            
            model_probs = []
            with torch.no_grad():
                for raw, gnn, _ in tqdm(holdout_loader, desc=f"Predicting with {name} model"):
                    logits = classifier(raw.to(self.device), gnn.to(self.device))
                    # Simplified: taking mean of sequence predictions
                    model_probs.extend(torch.sigmoid(logits).mean(dim=1).cpu().numpy())
            all_probs.append(np.array(model_probs))
        
        final_probs = (all_probs[0] + all_probs[1]) / 2.0
        final_preds = (final_probs > 0.5).astype(int)

        out_path = os.path.join(cfg.SUBMISSIONS_PATH, "submission.csv")
        # Ensure length matches holdout set
        pd.DataFrame({"speech_prob": final_preds[:len(holdout_loader.dataset)]}).to_csv(out_path, index=False)
        print(f"--- ✅ Submission file saved to {out_path} ---")
