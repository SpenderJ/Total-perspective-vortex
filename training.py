# coding: utf-8

import numpy as np
import os
import mne


from CSP import CSP  # use my own CSP

from mne.io import concatenate_raws
from mne.datasets import eegbci
from mne import events_from_annotations
from mne.channels import make_standard_montage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from joblib import dump

DATA_DIR = "mne_data"
SUBJECTS = [42]
RUNS1 = [6, 10, 14]  # motor imagery: hands vs feet
RUNS2 = [4, 8, 12]  # motor imagery: left hand vs right hand


def fetch_events(data_filtered, tmin=-1., tmax=4.):
    event_ids = dict(T1=0, T2=1)
    events, _ = events_from_annotations(data_filtered, event_id=event_ids)
    picks = mne.pick_types(data_filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(data_filtered, events, event_ids, tmin, tmax, proj=True,
                        picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1]
    return labels, epochs


def filter_data(raw, montage=make_standard_montage('standard_1020')):
    data_filter = raw.copy()
    data_filter.set_montage(montage)
    data_filter.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')
    p = mne.viz.plot_raw(data_filter, scalings={"eeg": 75e-6})
    return data_filter


def prepare_data(raw, montage=make_standard_montage('standard_1020')):
    raw.rename_channels(lambda x: x.strip('.'))
    eegbci.standardize(raw)
    raw.set_montage(montage)

    # plot
    montage = raw.get_montage()
    p = montage.plot()
    p = mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})
    return raw


def fetch_data(raw_fnames, sfreq=None):
    dataset = []
    subject = []
    for i, f in enumerate(raw_fnames):
        if f.endswith(".edf") and int(f.split('R')[1].split(".")[0]) in RUNS1:
            subject_data = mne.io.read_raw_edf(os.path.join(f"{DATA_DIR}/{SUBJECTS[0]}", f), preload=True)
            if sfreq is None:
                sfreq = subject_data.info["sfreq"]
            if subject_data.info["sfreq"] == sfreq:
                subject.append(subject_data)
            else:
                break
    dataset.append(mne.concatenate_raws(subject))
    raw = concatenate_raws(dataset)
    return raw


def training():
    raw = filter_data(raw=prepare_data(raw=fetch_data(raw_fnames=os.listdir(f"{DATA_DIR}/{SUBJECTS[0]}"))))
    labels, epochs = fetch_events(filter_data(raw))

    epochs_data_train = epochs.get_data()
    lda_shrinkage = LDA(solver='lsqr', shrinkage='auto')
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    csp = CSP()

    clf = Pipeline([('CSP', csp), ('LDA', lda_shrinkage)])
    scores_ldashrinkage = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
    mean_scores_ldashrinkage, std_scores_ldashrinkage = np.mean(scores_ldashrinkage), np.std(scores_ldashrinkage)

    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("LDA SHRINKED Classification accuracy: %f / Chance level: %f" % (np.mean(scores_ldashrinkage), class_balance))
    print(f"Mean Score Model {mean_scores_ldashrinkage}")
    print(f"Std Score Model {std_scores_ldashrinkage}")

    # save pipeline
    clf = clf.fit(epochs_data_train, labels)
    dump(clf, "final_model.joblib")
    print("model saved to final_model.joblib")
    pass


if __name__ == '__main__':
    training()
