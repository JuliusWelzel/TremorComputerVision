import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from scipy import signal

from src.config import (dir_figdata, dir_figures)
from src.MpHandAnalyst import MpHandAnalyst
from src.utls import calculate_amplitudes, calculate_displacement

# turn f_list into list of strings

# start analysing mp_objeo
mp_obj = MpHandAnalyst()
mp_obj.process_video(r"..\figdata\amp_lin_increase_30s.avi")
mp_obj.find_roi()

ground_truth_amp = pd.read_csv(dir_figdata.joinpath("sim_amp_lin_ground_truth.csv"), header=None)
ground_truth_amp.columns = ["frame", "amplitude"]
ground_truth_amp["times"] = ground_truth_amp["frame"] / 30

# get indices of the middle finger (12x, 12y, 12z)
idx_oi_mp = [list(mp_obj.cfg_mp_labels).index(label_oi) for label_oi in ["12x", "12y"]]
idx_oi_mp_z = [list(mp_obj.cfg_mp_labels).index(label_oi) for label_oi in ["12x", "12y", "12z"]]

# set the number of frames to drop from the end of the video
n_frames_drop = 2

# estimate magnitude of the middle finger amplitude
mp_world_amp_sum = np.sum(mp_obj.mp_positions_world[:-n_frames_drop, idx_oi_mp, 0], axis=1)
mp_norm_amp_sum_z = np.sum(mp_obj.mp_positions_norm[:-n_frames_drop, idx_oi_mp_z, 0], axis=1)
mp_norm_amp_sum = np.sum(mp_obj.mp_positions_norm[:-n_frames_drop, idx_oi_mp, 0], axis=1)

# estimate amplitudes and displacements
amplitudes_mp_world = calculate_amplitudes(mp_obj.mp_positions_world[:-n_frames_drop, idx_oi_mp, 0].T) # point 12 x,y only (maybe z)
displacment_mp_world = calculate_displacement(amplitudes_mp_world, 30) # this blender video is 30 fps

amplitudes_mp_norm_z = calculate_amplitudes(mp_obj.mp_positions_norm[:-n_frames_drop, idx_oi_mp_z, 0].T) # point 12 x,y only (maybe z)
displacment_mp_norm_z = calculate_displacement(amplitudes_mp_norm_z, 30) # this blender video is 30 fps

amplitudes_mp_norm = calculate_amplitudes(mp_obj.mp_positions_norm[:-n_frames_drop, idx_oi_mp, 0].T) # point 12 x,y only (maybe z)
displacment_mp_norm = calculate_displacement(amplitudes_mp_norm, 30) # this blender video is 30 fps

times_mp = np.linspace(0, len(amplitudes_mp_world)/30, len(amplitudes_mp_world) + 1)

# plot the amplitude over frame
fig, ax = plt.subplots(3, 1, figsize=(20, 15), sharex=True)

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

# plot the frequency over time
fig, ax = plt.subplots(3,1,figsize=(20, 10), sharex=True)

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

