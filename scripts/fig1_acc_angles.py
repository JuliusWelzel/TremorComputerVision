import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
import pandas as pd

# define paths
from src.config import (dir_figdata, dir_figures, setup_plt)

setup_plt(plt)

#set blender configurations
n_angles_side = 13
n_angles_total = 13 * 13
n_frames_per_angle = 180
n_chs_mp = 63

# process the angle data
acc_z = np.load(Path.joinpath(dir_figdata,'sim_acc_angles_blender.npy'))

# Reshape the array
acc_z_ang = acc_z.reshape(n_angles_total, n_frames_per_angle)

# calulate the mean for the accs per angle
acc_means = np.nanmean(acc_z_ang, axis=1)
acc_grid_means = acc_means.reshape((n_angles_side,n_angles_side))

# convole the data
kernel = Gaussian2DKernel(x_stddev=.5)
conv = convolve(acc_grid_means, kernel)
angles = np.linspace(-90,90,n_angles_side)

fig, axs = plt.subplots(figsize=(10,10))
image = axs.imshow(conv)

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
image.set_clim(0.4, 1)
# Set the color bar label
cbar.set_label('Mediapipe accuracy')
# increase the font size of the axes and colorbar
axs.tick_params(axis='both', which='major')

# save figure
plt.savefig(dir_figures.joinpath('fig1a_acc_mp_angles.png'), dpi = 600)




# process the angle data
acc_z = np.load(Path.joinpath(dir_figdata,'sim_acc_angles_apple.npy'))

# Reshape the array
acc_z_ang = acc_z.reshape(n_angles_total, n_frames_per_angle)

# calulate the mean for the accs per angle
acc_means = np.nanmean(acc_z_ang, axis=1)
acc_grid_means = acc_means.reshape((n_angles_side,n_angles_side))

# convole the data
kernel = Gaussian2DKernel(x_stddev=.5)
conv = convolve(acc_grid_means, kernel)
angles = np.linspace(-90,90,n_angles_side)

fig, axs = plt.subplots(figsize=(10,10))
image = axs.imshow(conv)

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
image.set_clim(0.4, 1)

# Set the color bar label
cbar.set_label('Apple CV accuracy')
# increase the font size of the axes and colorbar
axs.tick_params(axis='both', which='major')

# save figure
plt.savefig(dir_figures.joinpath('fig1b_acc_apple_angles.png'), dpi = 600)
