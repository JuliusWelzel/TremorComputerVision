"""
===========
Config file
===========

Configuration parameters for the study.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from socket import getfqdn
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
rec_colors = plt.cm.viridis(np.linspace(0, 1, 8))
rec_colors = rec_colors[[0,4,7],:]
prep_colors = plt.cm.magma(np.linspace(0, 1, 7))[[1,3,5],:]
cfg_colors = {"recording_type_colors": rec_colors, 
            "processing_type_colors": prep_colors,
            "mediapipe_color": "#ab6c82",
            "apple_color": "#f4d35e",}

###############################################################################
# Set size for figure text
cfg_ax_font = 16
cfg_title_font = 20
cfg_label_font = 12
cfg_legend_font = 12

###############################################################################
# Set standard font to use
cfg_font = "Times New Roman"