# LibriBrain-BCI  
**NeurIPS 2025 PNPL Competition — Brain–Computer Interface Models for Speech Detection & Phoneme Classification**  

This repository contains our solutions for the [LibriBrain Competition (NeurIPS 2025)](https://neurips.cc/), organized under the **PNPL Challenge**. The competition focuses on decoding language from **magnetoencephalography (MEG)** recordings, representing one of the largest publicly accessible non-invasive neuroimaging datasets for speech and language.  

Our project addresses both tasks of the competition:  

1. **Speech Detection (Task 1)** – Train a model to distinguish **speech vs. silence** from MEG activity recorded during listening. (**Implemented now**)  
2. **Phoneme Classification (Task 2)** – Build classifiers that map MEG activity to **specific phonemes** being heard. (**To be added later**)  

By combining **modern deep learning methods** with careful preprocessing of MEG signals, our aim is to contribute to the long-term vision of **non-invasive brain–computer interfaces (BCIs)** that can restore or augment human communication.  

---

## Research Motivation  

Decoding speech from brain activity has profound implications:  
- **Clinical impact:** enabling communication for individuals with speech-related disabilities.  
- **Scientific value:** advancing our understanding of the neural basis of language.  
- **Technological frontier:** driving progress in next-generation human–computer interaction.  

Unlike invasive BCI datasets (e.g., ECoG), the **LibriBrain MEG dataset** is large-scale and non-invasive (25–50× deeper than most prior datasets), lowering the barrier of entry for the broader ML and neuroscience communities.  

---

## Repository Structure  

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/<your-username>/LibriBrain-BCI.git
cd LibriBrain-BCI
pip install -r requirements.txt
````

We rely on the PNPL toolkit:
```bash
pip install pnpl
````
---

## Task 1: Speech Detection

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
