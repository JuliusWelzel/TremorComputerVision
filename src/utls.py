import numpy as np
from scipy import signal
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.stats import zscore
from scipy.stats import kendalltau
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import cfg_bandpass_fmin, cfg_bandpass_fmax



def calculate_amplitudes(raw_data):
    """
    Calculate the amplitude (distance) between consecutive points in a 3D space.

    Parameters:
    raw_data (np.array): An array of points where each row is a point (x, y, z).

    Returns:
    np.array: An array of amplitudes (distances) between consecutive points.
    """
    # Fill the missing values (NaNs) with the previous values
    raw_data = np.array(pd.DataFrame(raw_data.T).bfill()).T

    # check if data is in the right format
    if raw_data.shape[0] > raw_data.shape[1]:
        print("Warning: The input array should have less rows than columns (channels x samples).")

    # Calculate the differences between consecutive points
    differences = np.diff(raw_data, axis=1)

    # Calculate the Euclidean distances (amplitudes)
    amplitudes = np.sqrt(np.sum(differences**2, axis=0))

    return amplitudes


def calculate_displacement(amplitudes, srate = 30):
    """
    Calculates the displacement of the hand over time by integrating the
    velocity over time. The velocity is calculated by taking the derivative
    of the amplitude signal.

    Parameters
    ----------
    amplitudes : array_like
        The amplitude signal of the hand over time.
    srate : int
        The sampling rate of the amplitude signal.

    Returns
    -------
    displacements : array_like
        The displacement of the hand over time.

    """

    # Bandpass filter design parameters
    lowcut = cfg_bandpass_fmin  # low frequency cut-off in Hz
    highcut = cfg_bandpass_fmax  # high frequency cut-off in Hz

    # Normalizing the frequencies by the Nyquist frequency (half the sampling rate)
    nyq = 0.5 * srate
    low = lowcut / nyq
    high = highcut / nyq

    # Creating the Butterworth bandpass filter
    b, a = signal.butter(N=2, Wn=[low, high], btype='band')

    # Applying the filter to the signal
    filtered_signal = signal.filtfilt(b, a, np.array(amplitudes))

    # Performing peak detection on the filtered signal
    filtered_peaks, _ = signal.find_peaks(-filtered_signal)

    total_displacements = []

    for k in range(len(filtered_peaks) - 1):
        start_peak = filtered_peaks[k]
        end_peak = filtered_peaks[k + 1]
        displacement_sum = np.sum(np.abs(amplitudes[start_peak:end_peak]))
        total_displacements.append(displacement_sum)

    # Averaging the total displacements to get the amplitude
    average_total_displacement = np.nanmedian(total_displacements)

    return average_total_displacement


def mp_hand_labels():
    """
    Input : None

    Returns
    -------
    mp_labels : List of strings
        MediaPipe labels 0-63.

    """

    handMarks = np.arange(0, 21)  # number of tracked points
    dims = ["x", "y", "z"]  # dimensions in which hand is tracked
    labels = [str(h) + d for h in handMarks for d in dims]
    mp_labels = list(labels)
    return mp_labels
