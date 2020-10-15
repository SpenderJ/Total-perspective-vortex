# coding: utf-8

import os
import mne


def safe_opener(file):
    """
    Function used to safely open the csv file
    :param file: name of the file
    :return data: containing the training set
    """
    cwd = os.getcwd()
    try:
        f = open(os.path.join(cwd, file), 'rb')
        raw = mne.io.read_raw_edf(os.path.join(cwd, file), preload=True)
    except Exception as e:
        print("Cant open the EDF file passed as argument :" + file)
        raise e
    return raw
