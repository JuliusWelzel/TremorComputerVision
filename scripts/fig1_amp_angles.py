import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
import pandas as pd

# define paths
from src.config import (dir_figdata, dir_figures, setup_plt)
from src.utls import calculate_amplitudes, calculate_displacement, mp_hand_labels

setup_plt(plt)

#set blender configurations
n_angles_side = 13
n_angles_total = 13 * 13
n_frames_per_angle = 180
n_chs_mp = 63

# set labels for blender data
hand_labels = mp_hand_labels()
# get indices of the middle finger (12x, 12y, 12z)
idx_label_oi = [hand_labels.index(label_oi) for label_oi in ["12x", "12y"]]


# process the angle data
pos_z = np.load(Path.joinpath(dir_figdata,'sim_pos_angles_blender.npy'))

pos_z_ang = pos_z.reshape(n_angles_total, n_frames_per_angle, n_chs_mp)

# Apply the function on the second dimension of my_array

displament_per_angle = []
for i in range(n_angles_total):
    tmp_amp = calculate_amplitudes(pos_z_ang[i, 1:, idx_label_oi])
    tmp_displacement = calculate_displacement(tmp_amp, 30)
    displament_per_angle.append(tmp_displacement)

displacment_per_angle = np.reshape(displament_per_angle, (n_angles_side, n_angles_side))

angles = np.linspace(-90,90,n_angles_side)

fig, axs = plt.subplots(figsize=(10,10))
image = axs.imshow(displacment_per_angle, cmap='plasma')

# Set the x-axis tick labels
axs.set_xticks(np.arange(len(angles)))
axs.set_xticklabels(angles.astype(int))
# Set the y-axis tick labels
axs.set_yticks(np.arange(len(angles)))
axs.set_yticklabels(angles.astype(int))
axs.set_ylabel('Camera angle y-axis [°]')
axs.set_xlabel('Camera angle x-axis [°]')

# add labels 1,2,3 to the plots at (-40,-45), (0,0), (30,-60)
axs.text(2, 4, '1', ha='center', va='center', color='black', fontsize=20)
axs.text(6, 6, '2', ha='center', va='center', color='black', fontsize=20)
axs.text(3, 8, '3', ha='center', va='center', color='black', fontsize=20)

# Add color bar
cbar = fig.colorbar(image)

# Set the color bar label
cbar.set_label('Mediapipe displacement')
# increase the font size of the axes and colorbar
axs.tick_params(axis='both', which='major')

# save figure
plt.savefig(dir_figures.joinpath('fig1c_amp_mp_angles.png'), dpi = 600)


# process the angle data
pos_z = np.load(Path.joinpath(dir_figdata,'sim_pos_angles_apple.npy'))

pos_z_ang = pos_z.reshape(n_angles_total, n_frames_per_angle, n_chs_mp)

# Apply the function on the second dimension of my_array

displament_per_angle = []
for i in range(n_angles_total):
    tmp_amp = calculate_amplitudes(pos_z_ang[i, 1:, idx_label_oi])
    tmp_displacement = calculate_displacement(tmp_amp, 30)
    displament_per_angle.append(tmp_displacement)

displacment_per_angle = np.reshape(displament_per_angle, (n_angles_side, n_angles_side))

angles = np.linspace(-90,90,n_angles_side)

fig, axs = plt.subplots(figsize=(10,10))
image = axs.imshow(displacment_per_angle, cmap='plasma')

# Set the x-axis tick labels
axs.set_xticks(np.arange(len(angles)))
axs.set_xticklabels(angles.astype(int))
# Set the y-axis tick labels
axs.set_yticks(np.arange(len(angles)))
axs.set_yticklabels(angles.astype(int))
axs.set_ylabel('Camera angle y-axis [°]')
axs.set_xlabel('Camera angle x-axis [°]')

# add labels 1,2,3 to the plots at (-40,-45), (0,0), (30,-60)
axs.text(2, 4, '1', ha='center', va='center', color='black', fontsize=20)
axs.text(6, 6, '2', ha='center', va='center', color='black', fontsize=20)
axs.text(3, 8, '3', ha='center', va='center', color='black', fontsize=20)

# Add color bar
cbar = fig.colorbar(image)

# Set the color bar label
cbar.set_label('Apple Vision displacement')
# increase the font size of the axes and colorbar
axs.tick_params(axis='both', which='major')

# save figure
plt.savefig(dir_figures.joinpath('fig1d_amp_apple_angles.png'), dpi = 600)

