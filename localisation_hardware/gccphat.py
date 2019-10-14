import numpy as np
import constants


def gcc_phat(signal, reference_signal, fs=8000, max_tau=constants.max_tau, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = signal.shape[0] + reference_signal.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIGNAL = np.fft.rfft(signal, n=n)
    REFERENCE_SIGNAL = np.fft.rfft(reference_signal, n=n)
    R = SIGNAL * np.conj(REFERENCE_SIGNAL)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return tau, cc


def tdoa(signal, reference_signal, fs):
    tau, _ = gcc_phat(signal, reference_signal, fs=fs)
    theta = -np.arcsin(tau / constants.max_tau) * 180 / np.pi + 90
    return theta



