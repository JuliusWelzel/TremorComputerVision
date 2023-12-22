import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
pos_z=np.array(pd.DataFrame(pos_z.T).bfill(axis=0).T)
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
image = axs.imshow(np.log(displacment_per_angle), cmap='magma')

# Set the x-axis tick labels
axs.set_xticks(np.arange(len(angles)))
axs.set_xticklabels(angles.astype(int))
# Set the y-axis tick labels
axs.set_yticks(np.arange(len(angles)))
axs.set_yticklabels(angles.astype(int))
axs.set_ylabel('Camera angle y-axis [°]')
axs.set_xlabel('Camera angle x-axis [°]')

# add labels 1,2,3 to the plots at (-40,-45), (0,0), (30,-60)
axs.text(3, 2, '1', ha='center', va='center', color='black', fontsize=20)
axs.text(6, 6, '2', ha='center', va='center', color='black', fontsize=20)
axs.text(2, 8, '3', ha='center', va='center', color='black', fontsize=20)

# Add color bar
cbar = fig.colorbar(image)

# Set the color bar label
cbar.set_label('log(Mediapipe displacement)')
# increase the font size of the axes and colorbar
axs.tick_params(axis='both', which='major')

# save figure
plt.savefig(dir_figures.joinpath('fig1c_amp_mp_norm_angles.png'), dpi = 600)
plt.show()




idx_label_oi = [hand_labels.index(label_oi) for label_oi in ["12x", "12y","12z"]]


# process the angle data
pos_z = np.load(Path.joinpath(dir_figdata,'sim_pos_angles_blender.npy'))
pos_z=np.array(pd.DataFrame(pos_z.T).bfill(axis=0).T)
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
image = axs.imshow(np.log(displacment_per_angle), cmap='magma')

# Set the x-axis tick labels
axs.set_xticks(np.arange(len(angles)))
axs.set_xticklabels(angles.astype(int))
# Set the y-axis tick labels
axs.set_yticks(np.arange(len(angles)))
axs.set_yticklabels(angles.astype(int))
axs.set_ylabel('Camera angle y-axis [°]')
axs.set_xlabel('Camera angle x-axis [°]')

# add labels 1,2,3 to the plots at (-40,-45), (0,0), (30,-60)
axs.text(3, 2, '1', ha='center', va='center', color='black', fontsize=20)
axs.text(6, 6, '2', ha='center', va='center', color='black', fontsize=20)
axs.text(2, 8, '3', ha='center', va='center', color='black', fontsize=20)

# Add color bar
cbar = fig.colorbar(image)

# Set the color bar label
cbar.set_label('log(Mediapipe displacement)')
# increase the font size of the axes and colorbar
axs.tick_params(axis='both', which='major')

# save figure
plt.savefig(dir_figures.joinpath('fig1c_amp_mp_norm_z_angles.png'), dpi = 600)
plt.show()


idx_label_oi = [hand_labels.index(label_oi) for label_oi in ["12x", "12y"]]

# process the angle data
pos_z = np.load(Path.joinpath(dir_figdata,'sim_pos_angles_apple.npy'))

pos_z_ang = pos_z.reshape(n_angles_total, n_frames_per_angle, n_chs_mp)

# Apply the function on the second dimension of my_array
displament_per_angle = []
for i in range(n_angles_total):
    tmp_amp = calculate_amplitudes(pos_z_ang[i, 1:, idx_label_oi])
    tmp_displacement = calculate_displacement(tmp_amp, 30)
    displament_per_angle.append(tmp_displacement)

displacement_per_angle = np.reshape(displament_per_angle, (n_angles_side, n_angles_side))

fig, axs = plt.subplots(figsize=(10,10))
image = axs.imshow(displacement_per_angle, cmap='magma')

# Set the x-axis tick labels
axs.set_xticks(np.arange(len(angles)))
axs.set_xticklabels(angles.astype(int))
# Set the y-axis tick labels
axs.set_yticks(np.arange(len(angles)))
axs.set_yticklabels(angles.astype(int))
axs.set_ylabel('Camera angle y-axis [°]')
axs.set_xlabel('Camera angle x-axis [°]')

axs.text(3, 2, '1', ha='center', va='center', color='white', fontsize=20)
axs.text(6, 6, '2', ha='center', va='center', color='white', fontsize=20)
axs.text(2, 8, '3', ha='center', va='center', color='white', fontsize=20)


# Add color bar
cbar = fig.colorbar(image)
image.set_clim(0, 70) ## set lim to avoid noise at the edges.

# Set the color bar label
cbar.set_label('log(Vision displacement)')
# increase the font size of the axes and colorbar
axs.tick_params(axis='both', which='major')

# save figure
plt.savefig(dir_figures.joinpath('fig1d_amp_apple_angles.png'), dpi = 600)
plt.show()
