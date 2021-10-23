# coding: utf-8

import numpy as np
import os
import mne
import matplotlib.pyplot as plt

from CSP import CSP  # use my own CSP

from mne.io import concatenate_raws
from mne.datasets import eegbci
from mne import events_from_annotations
from mne.channels import make_standard_montage
# from mne.decoding import CSP  # use mne CSP
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from joblib import dump

data = 'mne_data'
tmin, tmax = -1., 4.
event_ids=dict(T1=0, T2=1)
subjects = [2]  # Use data of subject number 1
R1 = [6, 10, 14]  # motor imagery: hands vs feet
R2 = [4, 8, 12]  # motor imagery: left hand vs right hand

raw_fnames = os.listdir(f"{data}/{subjects[0]}")
dataset = []
subject = []
sfreq = None

for i, f in enumerate(raw_fnames):
    if f.endswith(".edf") and int(f.split('R')[1].split(".")[0]) in R2:
        subject_data = mne.io.read_raw_edf(os.path.join(f"{data}/{subjects[0]}", f), preload=True)
        if sfreq is None:
            sfreq = subject_data.info["sfreq"]
        if subject_data.info["sfreq"] == sfreq:
            subject.append(subject_data)
        else:
            break
dataset.append(mne.concatenate_raws(subject))
raw = concatenate_raws(dataset)

print(raw)
print(raw.info)
print(raw.info["ch_names"])
print(raw.annotations)

raw.rename_channels(lambda x: x.strip('.'))
montage = make_standard_montage('standard_1020')
eegbci.standardize(raw)

# create 10-05 system
raw.set_montage(montage)

# plot
montage = raw.get_montage()
p = montage.plot()
p = mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})

# data filtered
data_filter = raw.copy()
data_filter.set_montage(montage)
data_filter.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')
p = mne.viz.plot_raw(data_filter, scalings={"eeg": 75e-6})

# get events
events, _ = events_from_annotations(data_filter, event_id=event_ids)
picks = mne.pick_types(data_filter.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
epochs = mne.Epochs(data_filter, events, event_ids, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True)
labels = epochs.events[:, -1]

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


# Prediction

pivot = int(0.5 * len(epochs_data_train))

clf = clf.fit(epochs_data_train[:pivot], labels[:pivot])
try :
    p = clf.named_steps["CSP"].plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
except AttributeError:
    print("Method not implemented")

print("X shape= ", epochs_data_train[pivot:].shape, "y shape= ", labels[pivot:].shape)

scores = []
for n in range(epochs_data_train[pivot:].shape[0]):
    pred = clf.predict(epochs_data_train[pivot:][n:n + 1, :, :])
    print("n=", n, "pred= ", pred, "truth= ", labels[pivot:][n:n + 1])
    scores.append(1 - np.abs(pred[0] - labels[pivot:][n:n + 1][0]))
print("Mean acc= ", np.mean(scores))
pass