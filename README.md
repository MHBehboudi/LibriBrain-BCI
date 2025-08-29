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

- **Macro F1 Score:** 0.82  

## Task 2: Phoneme Classification (Coming Soon)

This task extends Task 1 by classifying which phoneme is being heard from the MEG recording:

```bash
fθ(X_MEG) → P = {aa, ae, ah, ...}
```

Implementation details and experiments will be added in Phase 2 of the project.
...
