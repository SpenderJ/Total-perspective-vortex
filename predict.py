# coding: utf-8

import numpy as np
import os
import mne

from joblib import load
from training import *

DATA_DIR = "mne_data"
PREDICT_MODEL = "final_model.joblib"
SUBJECTS = [42]


def predict():
    try:
        clf = load(PREDICT_MODEL)
    except FileNotFoundError as e:
        raise Exception(f"File not found: {PREDICT_MODEL}")

    # Fetch Data
    raw = filter_data(raw=prepare_data(raw=fetch_data(raw_fnames=os.listdir(f"{DATA_DIR}/{SUBJECTS[0]}"))))
    labels, epochs = fetch_events(filter_data(raw))
    epochs = epochs.get_data()

    print("X shape= ", epochs.shape, "y shape= ", labels.shape)

    scores = []
    for n in range(epochs.shape[0]):
        pred = clf.predict(epochs[n:n + 1, :, :])
        print("pred= ", pred, "truth= ", labels[n:n + 1])
        scores.append(1 - np.abs(pred[0] - labels[n:n + 1][0]))
    print("Mean acc= ", np.mean(scores))


if __name__ == "__main__":
    predict()
