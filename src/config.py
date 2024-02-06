"""
===========
Config file
===========

Configuration parameters for the study.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from socket import getfqdn
import seaborn as sns
from pathlib import Path


def define_dir(root, name):
    """define_dir create path handle and creates dir if not existend.


    Args:
        root (str): root to directory of interest
        name (str): name of directory

    Returns:
        pathlib.Path: Pathlib handle of dir
    """
    dir_oi = Path.joinpath(root, name)  # create path
    Path.mkdir(dir_oi, parents=False, exist_ok=True)
    return dir_oi


###############################################################################
# Determine which user is running the scripts on which machine and set the path
# where the data is stored and how many CPU cores to use.

user = os.getlogin()  # Username of the user running the scripts
host = getfqdn()  # Hostname of the machine running the scripts

if user == "User":
    # Julius Workstation
    dir_proj = Path(r"D:\kiel\TremorCv_elife")
elif user == "juliu":
    # Julius Laptop
    dir_proj = Path(r"C:\Users\juliu\Desktop\kiel\TremorCv_elife")



###############################################################################
# These are relevant directories which are used in the analysis.

# (import) helper functions
dir_sourcedata = Path.joinpath(dir_proj, "source")
dir_figdata = define_dir(dir_proj, "figdata")
dir_figures = define_dir(dir_proj, "figures")

###############################################################################
# These are all the relevant parameters for the analysis. You can experiment
# with changing these.

# Band-pass filter limits. Since we are performing ICA on the continuous data,
# it is important that the lower bound is at least 1Hz.
cfg_bandpass_fmin = 1  # Hz
cfg_bandpass_fmax = 10  # Hz

# Frequency window of interest in which the PCA show a peak
cfg_frequency_win_oi_min = 2  # Hz
cfg_frequency_win_oi_max = 8  # Hz

# Maximum number of PCA components consider
cfg_n_pca_components = 3  # n of PCs

###############################################################################
# These are all the relevant colors settings for the analysis
cv_models = {'MPnorm': '#006685', 'MPnorm_z': '#3FA5C4', 'MPworld': '#BF003F', 'MPworld_z': '#E84653', 'Apple_VI': '#FFE48D'}
cv_models_dict = {f"cv_model_{i+1}": color for i, color in enumerate(cv_models)}
rec_colors = plt.cm.viridis(np.linspace(0, 1, 8))
rec_colors = rec_colors[[0,4,7],:]
cfg_colors = {"recording_type_colors": rec_colors,
            "cv_model_colors": cv_models,
            "mediapipe_color": "#ab6c82",
            "apple_color": "#f4d35e",}

###############################################################################
# Set size for figure text
cfg_ax_font = 16
cfg_title_font = 22
cfg_label_font = 18
cfg_legend_font = 16

###############################################################################
# Set standard font to use
#cfg_font = "Open Sans"

def set_style(font_size: int = 24):
    """
    Just some basic things I do before plotting.
    """
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Open Sans Condensed'
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams.update({'font.size': cfg_title_font})
    mpl.rcParams.update({'axes.labelsize': cfg_ax_font})
    mpl.rcParams.update({'legend.fontsize': cfg_legend_font})
    mpl.rcParams.update({'xtick.labelsize': cfg_label_font})
    mpl.rcParams.update({'ytick.labelsize': cfg_label_font})
    mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'axes.spines.right': False})
    mpl.rcParams.update({'axes.spines.top': False})
