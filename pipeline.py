# coding: utf-8

import numpy as np
import os
import mne
import matplotlib.pyplot as plt

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne import Epochs, pick_types, find_events, events_from_annotations
from mne.channels import make_standard_montage
from mne.viz import *
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from joblib import dump

tmin, tmax = -1., 4.
event_ids=dict(hands=2, feet=3)   # 2 -> hands   | 3 -> feet
# subject = 1  # Use data of subject number 1
runs = [6, 10, 14]  # use only hand and feet motor imagery runs
raw_fnames = list()

for subject in [s for s in range(1, 2) if s not in (88, 92, 100)]: # We only use subject 1
    tmp_raw_fnames = eegbci.load_data(subject, runs)
    raw_fnames += tmp_raw_fnames

# Now globalize the data

raw = concatenate_raws([read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames])
raw.rename_channels(lambda x: x.strip('.'))

eegbci.standardize(raw)
# create 10-05 system
montage = make_standard_montage('standard_1005')
raw.set_montage(montage)
raw.filter(7., 30., method='iir')

events, _ = events_from_annotations(raw)

# picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

epochs = mne.Epochs(raw, events, event_ids, tmin, tmax, proj=True,
                        picks=picks, baseline=None, preload=True)

labels = epochs.events[:, -1] - 2

epochs_data_train = epochs.get_data()
cv = ShuffleSplit(10, test_size=0.2, random_state=42)


# Assemble a classifier
scores = []
lda = LDA()
lda_shrinkage = LDA(solver='lsqr', shrinkage='auto')
svc = SVC(gamma='auto')


csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)


clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores_lda = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
mean_scores_lda, std_scores_lda = np.mean(scores_lda), np.std(scores_lda)
clf = Pipeline([('CSP', csp), ('LDA', lda_shrinkage)])
scores_ldashrinkage = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
mean_scores_ldashrinkage, std_scores_ldashrinkage = np.mean(scores_ldashrinkage), np.std(scores_ldashrinkage)
clf = Pipeline([('CSP', csp), ('SVC', svc)])
scores_svc = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
mean_scores_svc, std_scores_svc = np.mean(scores_svc), np.std(scores_svc)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("LDA Classification accuracy: %f / Chance level: %f" % (np.mean(scores_lda), class_balance))
print("LDA SHRINKED Classification accuracy: %f / Chance level: %f" % (np.mean(scores_ldashrinkage), class_balance))
print("SVC Classification accuracy: %f / Chance level: %f" % (np.mean(scores_svc), class_balance))

sfreq = raw.info['sfreq']
w_length = int(sfreq * 0.5)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data_train.shape[2] - w_length, w_step)

scores_windows = []

for train_idx, test_idx in cv.split(epochs_data_train):
    print(1)
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx])

    # fit classifier
    lda_shrinkage.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        X_test = csp.transform(epochs_data_train[test_idx][:, :, n:(n + w_length)])
        score_this_window.append(lda_shrinkage.score(X_test, y_test))
    scores_windows.append(score_this_window)

# Plot scores over time
w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
plt.axvline(0, linestyle='--', color='k', label='Onset')
plt.axhline(0.5, linestyle='-', color='k', label='Chance')
plt.xlabel('time (s)')
plt.ylabel('classification accuracy')
plt.title('Classification score over time')
plt.legend(loc='lower right')
plt.show()

lda_shrinkage.fit(csp.fit_transform(epochs_data_train, labels), labels)
try:
    os.remove('model.joblib')
except OSError:
    pass
dump(lda_shrinkage, 'model.joblib')
pass