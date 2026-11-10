"""
A CNN Model for the Diagnosis of Schizophrenia from EEG Signals
Author: Farah MAALAOUI

Pipeline: load EEG (.edf) -> band-pass filter -> segment into windows
-> train 1D CNN -> evaluate (accuracy, precision, recall, F1, confusion matrix).

Expected data layout (e.g. RepOD dataset, 19 channels, 250 Hz):
    data/
      h01.edf, h02.edf, ...   # healthy controls  (file name starts with "h")
      s01.edf, s02.edf, ...   # schizophrenia     (file name starts with "s")
"""

import os
import glob
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- Parameters ----------------
DATA_DIR = "data"
SFREQ = 250          # sampling frequency (Hz)
WINDOW_SEC = 5       # window length (s)
L_FREQ, H_FREQ = 0.5, 45.0
EPOCHS = 50
BATCH_SIZE = 32
SEED = 42

np.random.seed(SEED)
keras.utils.set_random_seed(SEED)


# ---------------- 1. Load & preprocess ----------------
def load_subject(path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.filter(L_FREQ, H_FREQ, verbose=False)
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)
    data = raw.get_data()                          # (channels, samples)
    win = WINDOW_SEC * SFREQ
    n_win = data.shape[1] // win
    segments = [data[:, i * win:(i + 1) * win] for i in range(n_win)]
    segments = np.array(segments)                  # (n_win, channels, win)
    # z-score normalisation per segment and channel
    segments = (segments - segments.mean(axis=2, keepdims=True)) / (segments.std(axis=2, keepdims=True) + 1e-8)
    return segments


def load_dataset(data_dir):
    X, y, groups = [], [], []
    files = sorted(glob.glob(os.path.join(data_dir, "*.edf")))
    for subj_id, f in enumerate(files):
        label = 1 if os.path.basename(f).lower().startswith("s") else 0
        seg = load_subject(f)
        X.append(seg)
        y += [label] * len(seg)
        groups += [subj_id] * len(seg)
    X = np.concatenate(X).transpose(0, 2, 1)       # (n, time, channels)
    return X, np.array(y), np.array(groups)


# ---------------- 2. CNN model ----------------
def build_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, 7, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------- 3. Train & evaluate ----------------
def main():
    X, y, groups = load_dataset(DATA_DIR)
    print(f"Segments: {X.shape}, schizophrenia: {y.sum()}, healthy: {(y == 0).sum()}")

    # Subject-wise split: the same subject never appears in train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    model = build_model(X.shape[1:])
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
    )

    y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Schizophrenia"]))

    os.makedirs("results", exist_ok=True)

    # Training curves
    plt.figure(figsize=(10, 4))
    for i, m in enumerate(["accuracy", "loss"]):
        plt.subplot(1, 2, i + 1)
        plt.plot(history.history[m], label="train")
        plt.plot(history.history["val_" + m], label="validation")
        plt.title(m.capitalize()); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig("results/training_curves.png")

    # Confusion matrix
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=["Healthy", "Schizophrenia"]).plot(cmap="Blues")
    plt.savefig("results/confusion_matrix.png")

    model.save("results/cnn_schizophrenia.keras")


if __name__ == "__main__":
    main()
