import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from scipy.stats import zscore
from scipy import signal

from src.config import (dir_figdata, dir_figures,
                        cfg_frequency_win_oi_min, cfg_frequency_win_oi_max)
from src.MpHandAnalyst import MpHandAnalyst
from src.utls import calculate_amplitudes, calculate_displacement,calculate_displacement_over_time
import seaborn as sns
# turn f_list into list of strings

# start analysing mp_objeo
mp_obj = MpHandAnalyst()
mp_obj.process_video(r"figdata\amp_lin_increase_30s.avi")
mp_obj.find_roi()
freqs, specs, _ = mp_obj.frequency_estimations(to_plot=False)


ground_truth_amp = pd.read_csv(dir_figdata.joinpath("sim_amp_lin_ground_truth.csv"), header=None)
ground_truth_amp.columns = ["frame", "amplitude"]
ground_truth_amp["times"] = ground_truth_amp["frame"] / 30

# get indices of the middle finger (12x, 12y, 12z)
idx_oi_mp = [list(mp_obj.cfg_mp_labels).index(label_oi) for label_oi in ["12x", "12y"]]
idx_oi_mp_z = [list(mp_obj.cfg_mp_labels).index(label_oi) for label_oi in ["12x", "12y", "12z"]]

vi_data=pd.read_csv(r"figdata\sim_lin_amp_increase_appleCV.csv")[["middleTip_X","middleTip_Y"]]

# set the number of frames to drop from the end of the video
n_frames_drop = 2
# estimate magnitude of the middle finger amplitude
mp_world_amp_sum_z = np.sum(mp_obj.mp_positions_world[:-n_frames_drop, idx_oi_mp_z, 0], axis=1)
mp_world_amp_sum = np.sum(mp_obj.mp_positions_world[:-n_frames_drop, idx_oi_mp, 0], axis=1)
mp_norm_amp_sum_z = np.sum(mp_obj.mp_positions_norm[:-n_frames_drop, idx_oi_mp_z, 0], axis=1)
mp_norm_amp_sum = np.sum(mp_obj.mp_positions_norm[:-n_frames_drop, idx_oi_mp, 0], axis=1)
vi_amp_sum = np.sum(np.array(vi_data), axis=1)


# estimate amplitudes and displacements
amplitudes_mp_world = calculate_amplitudes(mp_obj.mp_positions_world[:-n_frames_drop, idx_oi_mp, 0].T) # point 12 x,y only (maybe z)s
displacment_mp_world = calculate_displacement(amplitudes_mp_world, 30) # this blender video is 30 fps
peak_times_world,ampltidudes_world=calculate_displacement_over_time(amplitudes_mp_world,30)

# estimate amplitudes and displacements
amplitudes_mp_world_z = calculate_amplitudes(mp_obj.mp_positions_world[:-n_frames_drop, idx_oi_mp_z, 0].T) # point 12 x,y only (maybe z)s
displacment_mp_world_z = calculate_displacement(amplitudes_mp_world, 30) # this blender video is 30 fps
peak_times_world_z,ampltidudes_world_z=calculate_displacement_over_time(amplitudes_mp_world,30)

amplitudes_mp_norm_z = calculate_amplitudes(mp_obj.mp_positions_norm[:-n_frames_drop, idx_oi_mp_z, 0].T) # point 12 x,y only (maybe z)
displacment_mp_norm_z = calculate_displacement(amplitudes_mp_norm_z, 30) # this blender video is 30 fps
peak_times_norm_z,ampltidudes_norm_z=calculate_displacement_over_time(amplitudes_mp_norm_z,30)

amplitudes_mp_norm = calculate_amplitudes(mp_obj.mp_positions_norm[:-n_frames_drop, idx_oi_mp, 0].T) # point 12 x,y only (maybe z)
displacment_mp_norm = calculate_displacement(amplitudes_mp_norm, 30) # this blender video is 30 fps
peak_times_norm,ampltidudes_norm=calculate_displacement_over_time(amplitudes_mp_norm,30)

amplitudes_vi = calculate_amplitudes(vi_data.T) # point 12 x,y only (maybe z)
displacment_vi = calculate_displacement(amplitudes_mp_norm, 30) # this blender video is 30 fps
peak_times_vi,ampltidudes_vision=calculate_displacement_over_time(amplitudes_vi,30)

colorblind = sns.color_palette("colorblind", 5)

times_mp = np.linspace(0, len(amplitudes_mp_world)/30, len(amplitudes_mp_world) + 1)
# plot the amplitude over frame




fig, ax = plt.subplots(5, 1, figsize=(15, 15),sharex=True)
ax[0].plot(ground_truth_amp["times"].iloc[0::2],-zscore(ground_truth_amp["amplitude"]).iloc[0::2],color="black",alpha=0.5,linestyle="dashed")
ax[0].plot(ground_truth_amp["times"].iloc[1::2],-zscore(ground_truth_amp["amplitude"]).iloc[1::2],color="black",alpha=0.5,linestyle="dashed")
ax[1].plot(ground_truth_amp["times"].iloc[0::2],-zscore(ground_truth_amp["amplitude"]).iloc[0::2],color="black",alpha=0.5,linestyle="dashed")
ax[1].plot(ground_truth_amp["times"].iloc[1::2],-zscore(ground_truth_amp["amplitude"]).iloc[1::2],color="black",alpha=0.5,linestyle="dashed")
ax[2].plot(ground_truth_amp["times"].iloc[0::2],-zscore(ground_truth_amp["amplitude"]).iloc[0::2],color="black",alpha=0.5,linestyle="dashed")
ax[2].plot(ground_truth_amp["times"].iloc[1::2],-zscore(ground_truth_amp["amplitude"]).iloc[1::2],color="black",alpha=0.5,linestyle="dashed")
ax[3].plot(ground_truth_amp["times"].iloc[0::2],-zscore(ground_truth_amp["amplitude"]).iloc[0::2],color="black",alpha=0.5,linestyle="dashed")
ax[3].plot(ground_truth_amp["times"].iloc[1::2],-zscore(ground_truth_amp["amplitude"]).iloc[1::2],color="black",alpha=0.5,linestyle="dashed")
ax[4].plot(ground_truth_amp["times"].iloc[0::2],-zscore(ground_truth_amp["amplitude"]).iloc[0::2],color="black",alpha=0.5,linestyle="dashed")
ax[4].plot(ground_truth_amp["times"].iloc[1::2],-zscore(ground_truth_amp["amplitude"]).iloc[1::2],color="black",alpha=0.5,linestyle="dashed")
ax[0].plot(times_mp,zscore(mp_world_amp_sum),color=colorblind[0],label="MP world")
ax[0].set_ylim(-3,3)
ax[0].legend(loc="upper left")
ax[1].plot(times_mp,zscore(mp_world_amp_sum_z),color=colorblind[1],label="MP world (with z axis)")
ax[1].set_ylim(-3,3)
ax[1].legend(loc="upper left")
ax[2].plot(times_mp,zscore(mp_norm_amp_sum),color=colorblind[2],label="MP norm")
ax[2].set_ylim(-3,3)
ax[2].legend(loc="upper left")
ax[3].plot(times_mp,zscore(mp_norm_amp_sum_z),color=colorblind[3],label="MP norm (with z axis)")
ax[3].set_ylim(-3,3)
ax[3].legend(loc="upper left")
ax[4].plot(times_mp,zscore(vi_amp_sum),color=colorblind[4],label="Vision")
ax[4].legend(loc="upper left")
ax[4].set_ylim(-3,3)
ax[4].set_xlabel("Time [s]")
ax[0].set_ylabel("Amplitude (z-scored)")
ax[1].set_ylabel("Amplitude (z-scored)")
ax[2].set_ylabel("Amplitude (z-scored)")
ax[3].set_ylabel("Amplitude (z-scored)")
ax[4].set_ylabel("Amplitude (z-scored)")

plt.tight_layout()
plt.legend()
fig.savefig(dir_figures.joinpath("amp_over_frames_alternative_new.png"), dpi=300)
plt.show()


srate=30
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot_normalized_spectrogram(signal_data, srate, ax, title, freq_range=None, num_segments=16):
    """
    Plots a normalized spectrogram on the given axes.

    :param signal_data: Array of signal data.
    :param srate: Sampling rate of the signal.
    :param ax: Matplotlib axes to plot on.
    :param title: Title for the subplot.
    :param freq_range: Tuple (min_freq, max_freq) to set y-axis limits.
    :param num_segments: Number of segments to normalize across.
    """
    # Compute the spectrogram
    f, t, Sxx = signal.spectrogram(signal_data, srate, nperseg=1 * srate)

    # Check if the spectrogram is large enough to be segmented
    if Sxx.shape[1] < num_segments:
        print(f"Warning: Not enough data points to divide into {num_segments} segments. Reducing number of segments.")
        num_segments = Sxx.shape[1]

    segment_length = Sxx.shape[1] // num_segments
    Sxx_normalized = np.zeros_like(Sxx)

    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        if np.any(Sxx[:, start:end]):  # Check if the segment is not empty
            segment_max = np.max(Sxx[:, start:end])
            segment_min = np.min(Sxx[:, start:end])
            Sxx_normalized[:, start:end] = (Sxx[:, start:end] - segment_min) / (segment_max - segment_min)

    # Plotting
    ax.pcolormesh(t, f, Sxx_normalized,cmap="viridis")
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title(title)
    if freq_range:
        ax.set_ylim(freq_range)

fig, ax = plt.subplots(10, 1, figsize=(20, 15), sharex=True)
ax[0].plot(times_mp,zscore(mp_world_amp_sum),color=colorblind[0],label="MP world")
ax[0].set_ylim(-3,3)
ax[0].legend()
plot_normalized_spectrogram(mp_world_amp_sum,srate,ax[1],"MP world")
ax[2].plot(times_mp,zscore(mp_world_amp_sum_z),color=colorblind[1],label="MP world (with z axis)")
ax[2].set_ylim(-3,3)
ax[2].legend()
plot_normalized_spectrogram(mp_world_amp_sum_z,srate,ax[3],"MP world (z-axis)")
ax[4].plot(times_mp,zscore(mp_norm_amp_sum),color=colorblind[2],label="MP norm")
ax[4].set_ylim(-3,3)
ax[4].legend()
plot_normalized_spectrogram(mp_norm_amp_sum,srate,ax[5],"MP norm")
ax[6].plot(times_mp,zscore(mp_norm_amp_sum_z),color=colorblind[3],label="MP norm (with z axis)")
ax[6].set_ylim(-3,3)
ax[6].legend()
plot_normalized_spectrogram(mp_norm_amp_sum_z,srate,ax[7],"MP norm (z-axis)")
ax[8].plot(times_mp,zscore(vi_amp_sum),color=colorblind[4],label="Vision")
ax[8].set_ylim(-3,3)
ax[8].legend()
plot_normalized_spectrogram(vi_amp_sum,srate,ax[9],"Vision")
plt.xlabel('Time [sec]')
plt.tight_layout()
# save the plot
plt.savefig(dir_figures.joinpath("freq_over_frames_alternative.png"), dpi=600)
plt.show()


fig, ax = plt.subplots(5, 1, figsize=(20, 15), sharex=True)
plot_normalized_spectrogram(mp_world_amp_sum,srate,ax[0],"MP world")
plot_normalized_spectrogram(mp_world_amp_sum_z,srate,ax[1],"MP world (z-axis)")
plot_normalized_spectrogram(mp_norm_amp_sum,srate,ax[2],"MP norm")
plot_normalized_spectrogram(mp_norm_amp_sum_z,srate,ax[3],"MP norm (z-axis)")
plot_normalized_spectrogram(vi_amp_sum,srate,ax[4],"Vision")
plt.xlabel('Time [sec]')
plt.tight_layout()
# save the plot
plt.xlim(0,5)
plt.savefig(dir_figures.joinpath("freq_over_frames_alternative2.png"), dpi=600)

plt.show()

# Example usage:a
plt.figure(figsize=(10, 8))

# Plotting multiple subplots
signals = [mp_world_amp_sum,mp_world_amp_sum_z, mp_norm_amp_sum, mp_norm_amp_sum_z, vi_amp_sum]  # replace with your actual signals
titles = ["MP world", "MP world (z-axis)", "MP norm", "MP norm (z-axis)","Vision"]
for i, (signal_data, title) in enumerate(zip(signals, titles)):
    ax = plt.subplot(5, 1, i + 1)  # Adjust subplot layout as needed
    plot_normalized_spectrogram(signal_data, srate, ax, title)

plt.tight_layout()
plt.show()





# plot the amplitude over frame
fig, ax = plt.subplots(4, 1, figsize=(20, 15), sharex=True)

# Increase fontsize of all text to 20
for a in ax:
    a.tick_params(axis='both', labelsize=12)
    a.set_title(a.get_title(), fontsize=20)
    a.set_xlabel(a.get_xlabel(), fontsize=14)
    a.set_ylabel(a.get_ylabel(), fontsize=14)
    a.legend(fontsize=14)

ax[0].plot(ground_truth_amp["times"], -zscore(ground_truth_amp["amplitude"]), label="Ground truth", color="grey")
ax[0].plot(times_mp, zscore(mp_world_amp_sum), label="Mediapipe world", color = "#ab6c82")
ax[0].set_ylabel("Amplitude (z-scored)")
ax[0].legend()
ax[0].set_title("Blender y-amplitude over frames")


ax[1].plot(ground_truth_amp["times"], -zscore(ground_truth_amp["amplitude"]), label="Ground truth", color="grey")
ax[1].plot(times_mp, zscore(mp_norm_amp_sum_z), label="Mediapipe norm", color = "#ab6c82")
ax[1].set_ylabel("Amplitud (z-scored)")
ax[1].legend()

ax[2].plot(ground_truth_amp["times"], -zscore(ground_truth_amp["amplitude"]), label="Ground truth", color="grey")
ax[2].plot(times_mp, zscore(mp_norm_amp_sum), label="Mediapipe norm (no-zaxis)", color = "#ab6c82")
ax[2].set_xlabel("Time [s]")
ax[2].set_ylabel("Amplitude (z-scored)")
ax[2].legend()

ax[2].plot(ground_truth_amp["times"], -zscore(ground_truth_amp["amplitude"]), label="Ground truth", color="grey")
ax[3].plot(times_mp, zscore(vi_amp_sum), label="Mediapipe norm (no-zaxis)", color = "#ab6c82")
ax[3].set_xlabel("Time [s]")
ax[3].set_ylabel("Amplitude (z-scored)")
ax[3].legend()

plt.tight_layout()
plt.show()

# save the plot
fig.savefig(dir_figures.joinpath("amp_over_frames.png"), dpi=600)

# plot the amplitude over frame
fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

# Increase fontsize of all text to 20
for a in ax:
    a.tick_params(axis='both', labelsize=12)
    a.set_title(a.get_title(), fontsize=20)
    a.set_xlabel(a.get_xlabel(), fontsize=14)
    a.set_ylabel(a.get_ylabel(), fontsize=14)
    a.legend(fontsize=14)

ax[0].plot(ground_truth_amp["times"], -zscore(ground_truth_amp["amplitude"]), label="Ground truth", color="grey")
ax[0].plot(times_mp, zscore(mp_world_amp_sum), label="Mediapipe world", color = "#ab6c82")
ax[0].set_ylabel("Amplitude (z-scored)")
ax[0].legend()
ax[0].set_title("Blender y-amplitude over frames")
ax[0].set_xlim(0, 3)


ax[1].plot(ground_truth_amp["times"], -zscore(ground_truth_amp["amplitude"]), label="Ground truth", color="grey")
ax[1].plot(times_mp, zscore(mp_norm_amp_sum_z), label="Mediapipe norm", color = "#ab6c82")
ax[1].set_ylabel("Amplitud (z-scored)")
ax[1].legend()
ax[1].set_xlim(0, 3)

ax[2].plot(ground_truth_amp["times"], -zscore(ground_truth_amp["amplitude"]), label="Ground truth", color="grey")
ax[2].plot(times_mp, zscore(mp_norm_amp_sum), label="Mediapipe norm (no-zaxis)", color = "#ab6c82")
ax[2].set_xlabel("Time [s]")
ax[2].set_ylabel("Amplitude (z-scored)")
ax[2].legend()
ax[2].set_xlim(0, 3)

plt.tight_layout()
plt.show()

# save the plot
fig.savefig(dir_figures.joinpath("amp_over_frames_firstSec.png"), dpi=600)

# get the frequency of the middle finger over time for each method
srate = 30
fw, tw, Sxxw = signal.spectrogram(mp_world_amp_sum, srate, nperseg=2 * srate, noverlap=srate)
fn, tn, Sxxn = signal.spectrogram(mp_norm_amp_sum, srate, nperseg=2 * srate, noverlap=srate)
fz, tz, Sxxz = signal.spectrogram(mp_norm_amp_sum_z, srate, nperseg=2 * srate, noverlap=srate)
fv, tv, Sxxv = signal.spectrogram(vi_amp_sum, srate, nperseg=2 * srate, noverlap=srate)
from matplotlib import colors
fig, ax = plt.subplots(4,1,figsize=(20, 10), sharex=True)

ax[0].pcolormesh(tw, fw, Sxxw,)
ax[0].set_ylabel('Frequency [Hz]')
ax[0].set_title("Mediapipe world")
ax[0].set_ylim(4, 8)
ax[1].pcolormesh(tz, fz, Sxxz )
ax[1].set_ylabel('Frequency [Hz]')
ax[1].set_title("Mediapipe norm (z-axis)")
ax[1].set_ylim(4, 8)
ax[2].pcolormesh(tn, fn, Sxxn, )
ax[2].set_ylabel('Frequency [Hz]')
ax[2].set_title("Mediapipe norm")
ax[2].set_xlabel('Time [sec]')
ax[2].set_ylim(4, 8)
ax[3].pcolormesh(tv, fv, Sxxv, )
ax[3].set_ylabel('Frequency [Hz]')
ax[3].set_title("Vision")
ax[3].set_xlabel('Time [sec]')
ax[3].set_ylim(4, 8)
# Increase fontsize of all text to 20
for a in ax:
    a.tick_params(axis='both', labelsize=12)
    a.set_title(a.get_title(), fontsize=20)
    a.set_xlabel(a.get_xlabel(), fontsize=14)
    a.set_ylabel(a.get_ylabel(), fontsize=14)
    a.legend(fontsize=14)
plt.tight_layout()
plt.show()

# save the plot
fig.savefig(dir_figures.joinpath("freq_over_frames.png"), dpi=600)



import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = 'figdata/sim_amp_lin_ground_truth.csv'
data = pd.read_csv(file_path)

# Assuming the CSV file has columns named 'Time' and 'Amplitude'
time = data.iloc[:, 0]  # First column as time
amplitude = data.iloc[:, 1]  # Second column as amplitude
amp_max=data.iloc[1::2]
amp_min=data.iloc[0::2]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(amp_max.iloc[:,0],amp_max.iloc[:,1])
plt.plot(amp_min.iloc[:,0],amp_min.iloc[:,1])
# Plot the signal
plt.plot(time, amplitude, label='Signal')

# Shade the area under the signal
plt.fill_between(time, amplitude, color='skyblue', alpha=0.4)

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Signal with Shaded Area')
plt.legend()

# Show the plot
plt.show()
