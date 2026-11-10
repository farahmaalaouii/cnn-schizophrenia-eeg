# A CNN Model for the Diagnosis of Schizophrenia from EEG Signals

Research Master's project, Master of Computer Science (option *Smart Pervasive Systems*), ISSAT Sousse, University of Sousse.

**Author:** Farah MAALAOUI
**Supervisor:** Dr. Kais BOUALLEGUE

---

## Overview

A compact Convolutional Neural Network (**140,225 parameters**) that diagnoses schizophrenia from resting-state EEG. The model is evaluated with strict **leave-one-subject-out** cross-validation, so every test subject is completely unseen during training. The project also shows that the usual random segment-wise split causes **data leakage** and strongly overestimates performance.

## Dataset

Public **RepOD** dataset (Institute of Psychiatry and Neurology, Warsaw):

- 14 patients with paranoid schizophrenia (ICD-10 F20.0) and 14 healthy controls (7 men and 7 women per group, about 28 years old, matched for sex and age)
- Resting state, eyes closed, about 15 minutes per subject
- 19 channels (10–20 system), 250 Hz
- **10,917 clean 5-second segments** after preprocessing (5,875 SZ, 5,042 HC)

The dataset is not included in this repository.

## Methodology

**1. Preprocessing**
- Band-pass filter 0.5–45 Hz and 50 Hz notch filter
- Common average reference
- Automatic removal of eye and muscle artifacts with ICA + ICLabel
- 5-second windows with 50% overlap
- Rejection of windows above 150 µV, then normalisation

**2. Model**: compact CNN inspired by EEGNet and Schirrmeister's networks
- Input: raw signal, 19 channels × 1,250 samples
- Temporal convolution (200 ms filters), then spatial convolution over the 19 channels
- Three deeper temporal blocks
- Batch normalisation, ELU activations, dropout
- Global average pooling and sigmoid output

**3. Training and evaluation**
- **Leave-one-subject-out cross-validation (28 folds)**: in each fold, 1 test subject, 2 subjects (1 patient + 1 control) for early stopping, 25 for training; the model is trained from scratch in every fold
- Adam (learning rate 0.001), binary cross-entropy, up to 40 epochs, early stopping (patience 5), data augmentation
- **Subject-level diagnosis**: the segment probabilities of each subject (about 390 segments) are averaged into one decision per person
- Metrics at subject level with 95% bootstrap confidence intervals

## Results

### Main result (leave-one-subject-out, subject level)

**24 of 28 unseen subjects correctly diagnosed.**

| Metric      | Value |
|-------------|-------|
| Accuracy    | **85.7%** (95% CI 71.4–96.4%) |
| Sensitivity | **85.7%** (12/14 patients) |
| Specificity | **85.7%** (12/14 controls) |
| AUC         | **0.918** |

Accuracy at segment level: 79.5%. Averaging per subject raises it to 85.7%.

### Comparison with baselines (same segments, same folds)

| Model                      | Accuracy   | Sensitivity | Specificity | AUC       |
|----------------------------|------------|-------------|-------------|-----------|
| SVM (band power)           | 75.00%     | 85.71%      | 64.29%      | 0.806     |
| Random forest (band power) | 71.43%     | 78.57%      | 64.29%      | 0.709     |
| **Proposed CNN**           | **85.71%** | **85.71%**  | **85.71%**  | **0.918** |

The best previous subject-wise result on RepOD is 81.26% (Oh et al., 2019). With only 28 subjects, the differences with the baselines are not statistically significant (p = 0.46 and 0.30).

### Effect of data leakage

| Validation protocol         | Segment accuracy |
|-----------------------------|------------------|
| Random segment-wise split   | 99.99% (1 error out of 10,917) |
| Leave-one-subject-out       | 79.51% |

The same model on the same data gains about 20 points with a random split. This happens because the model learns to recognise subjects instead of detecting the disease. Near-perfect results reported in the literature are therefore largely optimistic.

## Limitations

- Small dataset (28 subjects), hence wide confidence intervals
- Training on CPU with a single random seed
- Deep baselines and the ablation study are implemented but were not run (about 5 hours per run on CPU)

## Technologies

Python · Deep learning (CNN) · MNE (EEG preprocessing, ICA, ICLabel) · Scikit-learn · NumPy · Matplotlib

## License

This project is for academic purposes.
