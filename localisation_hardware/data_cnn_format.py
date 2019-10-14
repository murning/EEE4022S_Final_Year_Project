import numpy as np


def normalize_zero_mean(waveform):
    output = (waveform - np.mean(waveform)) / np.std(waveform)
    return output


def normalize_x_data(x_data):
    return np.asarray([normalize_zero_mean(i) for i in x_data])


def reshape_x_for_cnn(x_data):
    return np.asarray([np.array([x]).T for x in x_data])
