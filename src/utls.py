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
    amplitudes = np.sqrt(np.nansum(differences**2, axis=0))

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
        displacement_sum = np.nansum(np.abs(amplitudes[start_peak:end_peak]))
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


def find_handlandmarks(mp_class, tmp_image):
    """
    This functions finds the coordinates of all hand landmarks in a single image/frame
    """

    # prelocate vars
    tmp_pos_norm = np.zeros([len(mp_class.cfg_mp_labels), 2])
    tmp_pos_norm[:] = np.nan
    tmp_pos_world = np.zeros([len(mp_class.cfg_mp_labels), 2])
    tmp_pos_world[:] = np.nan
    tmp_acc = np.zeros([1, 2])
    label_hand = ["", ""]
    tmp_pos_norm_mp = []
    tmp_pos_world_mp = []

    # prep image via cv and process via MP (self.hands)
    tmp_image.flags.writeable = False
    tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
    results = mp_class.hands.process(tmp_image)

    if results.multi_hand_landmarks is not None:

        for ih, (hand_norm, hand_world) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks)
        ):  # results.multi_hand_landmarks returns normalised landMarks for all the hands

            # get hand classification label
            label_hand[ih] = results.multi_handedness[ih].classification[0].label
            tmp_pos_norm_mp = []
            tmp_pos_world_mp = []

            for landmark_norm, landmark_world in zip(
                hand_norm.landmark, hand_world.landmark
            ):
                # landMark holds x,y,z ratios of single landmark
                x_norm, y_norm, z_norm = (
                    landmark_norm.x,
                    landmark_norm.y,
                    landmark_norm.z,
                )
                tmp_pos_norm_mp.append([x_norm, y_norm, z_norm])
                x_world, y_world, z_world = (
                    landmark_world.x,
                    landmark_world.y,
                    landmark_world.z,
                )
                tmp_pos_world_mp.append([x_world, y_world, z_world])
                tmp_acc[:, ih] = results.multi_handedness[ih].classification[0].score

            tmp_pos_norm[:, ih] = np.asarray(tmp_pos_norm_mp).flatten()
            tmp_pos_world[:, ih] = np.asarray(tmp_pos_world_mp).flatten()

    landmark_position_norm = tmp_pos_norm
    landmark_position_world = tmp_pos_world
    hand_accuracy = tmp_acc

    return landmark_position_norm, landmark_position_world, hand_accuracy, label_hand


def pcs2spec(cfg_n_components, cfg_freqs_oi, specs, freqs):
    """This function takes a number of principal components and weights them
    to get a spectrum for a frequency window of interest.

    Args:
        cfg_n_components (int): Number of PCs to take into account for spectral analysis.
        cfg_freqs_oi (list of str): Frequency window in which a PC spectrum should have a peak.
        specs (ndarray): Array of type float.
        freqs (ndarray): Array of type float.

    Returns:
        ndarray: Array of type float.
    """
    peaks_freq_raw = []
    for i in range(cfg_n_components):
        peak, props = signal.find_peaks(specs[:, i], height=np.max(specs[:, i]) * 0.1)
        if not props["peak_heights"].any():
            continue
        idx_max = np.argmax(props["peak_heights"])
        peaks_freq_raw.append(freqs[peak][idx_max])

    peaks_freq_raw = np.array(peaks_freq_raw)
    idx_freqs_oi = np.where(
        np.logical_and(peaks_freq_raw >= cfg_freqs_oi[0], peaks_freq_raw <= cfg_freqs_oi[1])
    )
    idx_pcs_oi = idx_freqs_oi[0][:cfg_n_components]
    specs_oi = specs[:, idx_pcs_oi]


    if specs_oi.any():
        idx_peak_oi = np.argmax(specs_oi.max(axis=0))
        peaks_amp_raw_oi = np.max(specs_oi.max(axis=0))
        peaks_freq_raw_oi = peaks_freq_raw[idx_peak_oi]
        specs_oi = specs_oi[:,idx_peak_oi]
    else:
        peaks_amp_raw_oi = np.nan
        peaks_freq_raw_oi = np.nan

    return specs_oi, peaks_freq_raw_oi, peaks_amp_raw_oi


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a Butterworth bandpass filter.

    Parameters
    ----------
    lowcut : float
        The low cut-off frequency in Hz.
    highcut : float
        The high cut-off frequency in Hz.
    fs : int
        The sampling rate in Hz.
    order : int, optional
        The order of the filter. The default is 5.

    Returns
    -------
    b : array_like
        The numerator of the filter.
    a : array_like
        The denominator of the filter.

    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="bandpass", analog=False)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis = 0):
    """
    Apply a Butterworth bandpass filter to a signal.

    Parameters
    ----------
    data : array_like
        The signal to be filtered.
    lowcut : float
        The low cut-off frequency in Hz.
    highcut : float
        The high cut-off frequency in Hz.
    fs : int
        The sampling rate in Hz.
    order : int, optional
        The order of the filter. The default is 5.
    axis : int, optional
        The axis along which to filter. The default is 0.

    Returns
    -------
    y : array_like
        The filtered signal.

    """

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data, axis = axis)
    return y