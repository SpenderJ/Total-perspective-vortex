# coding: utf-8

import mne

from mne.datasets import eegbci # Corresponds to the EEGCBI motor imagery
from mne import Epochs, pick_types, find_events, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.viz import *
from mne.channels import make_standard_montage

dataset = 'eegmmidb/S001R01.edf'

# # Set parameters

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.

"""
                Set params
"""
tmin, tmax = -1., 4.
event_ids=dict(hands=2, feet=3)   # 2 -> hands   | 3 -> feet
subject = 1  # Use data of subject number 1
runs = [6, 10, 14]  # use only hand and feet motor imagery runs

def analyze():
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames])
    raw.rename_channels(lambda x: x.strip('.'))

    eegbci.standardize(raw)
    # create 10-05 system
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw.filter(7, 13)

    events, _ = events_from_annotations(raw)
    picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])
    epochs = mne.Epochs(raw, events, event_ids, tmin, tmax,
                        picks=picks, baseline=None, preload=True)

    plot_events(events)
    epochs.plot_psd()

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = Epochs(raw, events, event_ids, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs.plot_psd()



    pass


if __name__ == '__main__':
    analyze()