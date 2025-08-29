# Neural Decoding of Speech
**Brain–Computer Interface Models for Speech Detection & Phoneme Classification**  

This repository contains our solutions for decoding speech from **magnetoencephalography (MEG)** recordings.

Our project addresses two tasks:  

1. **Speech Detection** – Train a model to distinguish **speech vs. silence** from MEG activity recorded during listening. (**Implemented now**)  
2. **Phoneme Classification** – Build classifiers that map MEG activity to **specific phonemes** being heard. (**Planned — Phase 2**)  

By combining **modern deep learning methods** with careful preprocessing of MEG signals, our aim is to contribute to the long-term vision of **non-invasive brain–computer interfaces (BCIs)** that can restore or augment human communication.  

---

## Research Motivation  

Decoding speech from brain activity has profound implications:  
- **Clinical impact:** enabling communication for individuals with speech-related disabilities.  
- **Scientific value:** advancing our understanding of the neural basis of language.  
- **Technological frontier:** driving progress in next-generation human–computer interaction.  

Unlike invasive BCI datasets (e.g., ECoG), **MEG datasets** are non-invasive, lowering the barrier of entry for the broader ML and neuroscience communities.  

---

## Repository Structure

- [README.md](README.md)  
- [requirements.txt](requirements.txt)  
- [configs/](configs/) – Training & model configuration files  
- [data/](data/) – Dataset download & preprocessing scripts  
- [models/](models/) – Model architectures (CNNs, Transformers, GNNs, etc.)  
- [training/](training/) – Training loops & utilities  
- [evaluation/](evaluation/) – Metrics, threshold tuning, leaderboard submission  
- [task1_speech/](task1_speech/) – Speech detection experiments  
  - [notebooks/](task1_speech/notebooks/) – **Scripted pipeline (converted from notebooks)**
    - [0_setup_and_env.py](task1_speech/notebooks/0_setup_and_env.py)  
    - [1_pretrain_eeg2rep.py](task1_speech/notebooks/1_pretrain_eeg2rep.py)  
    - [2_train_speech_detection.py](task1_speech/notebooks/2_train_speech_detection.py)  
    - [3_validate_and_threshold.py](task1_speech/notebooks/3_validate_and_threshold.py)  
    - [4_ensemble_and_submission.py](task1_speech/notebooks/4_ensemble_and_submission.py)  
    - [5_utils_gnn_clustering.py](task1_speech/notebooks/5_utils_gnn_clustering.py)  
    - [README.md](task1_speech/notebooks/README.md)  
  - [scripts/](task1_speech/scripts/) – Refactored Python modules (optional)  
  - [results/](task1_speech/results/) – Logs, checkpoints, submissions  
- [task2_phoneme/](task2_phoneme/) – (empty, Phase 2)

---

## Installation

```bash
git clone https://github.com/<your-username>/LibriBrain-BCI.git
cd LibriBrain-BCI
pip install -r requirements.txt
# toolkit
pip install pnpl
````

---

## Task 1: Speech Detection
All experiments for Task 1 (speech vs. silence classification) are in:
Pipeline (scripts):

- [0_setup_and_env.py](task1_speech/notebooks/0_setup_and_env.py)   — Environment setup & config
- [1_pretrain_eeg2rep.py](task1_speech/notebooks/1_pretrain_eeg2rep.py)  — Self-supervised pretraining
- [2_train_speech_detection.py](task1_speech/notebooks/2_train_speech_detection.py) — Supervised fine-tuning
- [3_validate_and_threshold.py](task1_speech/notebooks/3_validate_and_threshold.py)  — Validation & threshold tuning
- [4_ensemble_and_submission.py](task1_speech/notebooks/4_ensemble_and_submission.py)  — Ensemble & submission generation
- [5_utils_gnn_clustering.py](task1_speech/notebooks/5_utils_gnn_clustering.py)  — Sensor clustering & GNN utilities

**Problem Definition:** Given an MEG recording segment, predict whether the subject was hearing speech or silence at that moment.

Formally:
```bash
fθ(X_MEG) → {0 = silence, 1 = speech}
````
where X_MEG denotes the multichannel MEG timeseries.

**Training:** 
```bash
python training/train_speech.py --config configs/speech.yaml
````
**Evaluation:** Generate predictions for submission:
```bash
python evaluation/eval_speech.py --input checkpoints/speech_model.pt
````
### Results (Task 1)

On the official competition test dataset provided by the organizers, our best model achieved:

- **Macro F1 Score:** 0.82  

## Task 2: Phoneme Classification (Coming Soon)

This task extends Task 1 by classifying which phoneme is being heard from the MEG recording:

```bash
fθ(X_MEG) → P = {aa, ae, ah, ...}
```

Implementation details and experiments will be added in Phase 2 of the project.

## Evaluation Protocol

- Shared holdout set provided by competition organizers.

- Metrics:

  - Speech Detection → Accuracy, F1 score

  - Phoneme Classification → Accuracy, Macro-F1, Confusion Matrix

## Research Directions

Some directions we are exploring:
- **Spatiotemporal deep learning:** Transformers, ConvNets, and temporal attention mechanisms for MEG decoding
- **Representation learning:** Contrastive and self-supervised pretraining on large MEG corpora
- **Cross-task generalization:** Transfer learning between speech detection and phoneme classification
- **Neuroscience alignment:** Comparing learned representations with known auditory/linguistic cortical hierarchies

## Project Pipeline

Below is the high-level workflow from MEG recordings to predictions:

[ MEG Data ] → [ Preprocessing ] → [ Deep Learning Model ] → [ Speech / Phoneme Prediction ]


(A detailed diagram can be added here: docs/pipeline.png)

## Contributions

We welcome contributions, feedback, and discussions! Please open an issue or pull request if you’d like to collaborate.

## References

[LibriBrain Competition Website (NeurIPS 2025)](https://neural-processing-lab.github.io/2025-libribrain-competition/tracks/)
