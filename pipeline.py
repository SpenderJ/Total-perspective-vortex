# coding: utf-8

import numpy as np
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

tmin, tmax = -1., 4.
event_ids=dict(hands=2, feet=3)   # 2 -> hands   | 3 -> feet
# subject = 1  # Use data of subject number 1
runs = [6, 10, 14]  # use only hand and feet motor imagery runs
raw_fnames = list()

for subject in [s for s in range(1, 2) if s not in (88, 92, 100)]:
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
cv_split = cv.split(epochs_data_train)


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
