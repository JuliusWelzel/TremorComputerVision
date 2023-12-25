from src.config import (cfg_ax_font, cfg_label_font, cfg_title_font, cfg_legend_font) # import font size
from src.config import (cfg_colors) # import colors
from src.config import (dir_figdata, dir_figures) # import figure directories

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
import numpy as np
from scipy.stats import kruskal
import scikit_posthocs as sp

amp_data=pd.read_csv(Path.joinpath(dir_figdata,"displacements.csv"))

# Calculation of amplitude estimation error relative to OMC in mm
amp_data["error_mp_world_displacement"]=np.abs(amp_data["mocap_displacement"]-amp_data["mp_world_displacement"])
amp_data["error_mp_world_z_displacement"]=np.abs(amp_data["mocap_displacement"]-amp_data["mp_world_z_displacement"])
amp_data["error_mp_norm_displacement"]=np.abs(amp_data["mocap_displacement"]-amp_data["mp_norm_displacement"])
amp_data["error_mp_norm_displacement_z"]=np.abs(amp_data["mocap_displacement"]-amp_data["mp_norm_displacement_z"])
amp_data["error_apple_displacement"]=np.abs(amp_data["mocap_displacement"]-amp_data["apple_displacement"])


# Calculation of amplitude estimation error relative to OMC in % of OMC amplitude
amp_data["perc_error_mp_world_displacement"]=np.abs((amp_data["mocap_displacement"]-amp_data["mp_world_displacement"]))/amp_data["mocap_displacement"]*100
amp_data["perc_error_mp_world_z_displacement"]=np.abs((amp_data["mocap_displacement"]-amp_data["mp_world_z_displacement"]))/amp_data["mocap_displacement"]*100
amp_data["perc_error_mp_norm_displacement"]=np.abs((amp_data["mocap_displacement"]-amp_data["mp_norm_displacement"]))/amp_data["mocap_displacement"]*100
amp_data["perc_error_mp_norm_displacement_z"]=np.abs((amp_data["mocap_displacement"]-amp_data["mp_norm_displacement_z"]))/amp_data["mocap_displacement"]*100
amp_data["perc_error_apple_displacement"]=np.abs((amp_data["mocap_displacement"]-amp_data["apple_displacement"]))/amp_data["mocap_displacement"]*100


amp_data = amp_data.drop('id', axis=1)

amp_abs=amp_data[["error_mp_world_displacement","error_mp_world_z_displacement","error_mp_norm_displacement","error_mp_norm_displacement_z","error_apple_displacement"]]
amp_abs.columns=["MP world","MP world z-axis","MP norm", "MP norm z-axis","VI"]
median_values=amp_abs.median()
quartiles=amp_abs.quantile([0.25,0.75]).T
summary_stats_abs = pd.concat([median_values, quartiles], axis=1)
summary_stats_abs.columns = ['Median', '25th Percentile', '75th Percentile']

summary_stats_abs.to_markdown(Path.joinpath(dir_figures,"suppl_table_1.md"))

amp_relative=amp_data[["perc_error_mp_world_displacement","perc_error_mp_world_z_displacement","perc_error_mp_norm_displacement","perc_error_mp_norm_displacement_z","perc_error_apple_displacement"]]
amp_relative.columns=["MP world","MP world z-axis","MP norm", "MP norm z-axis","VI"]
median_values=amp_relative.median()
quartiles=amp_relative.quantile([0.25,0.75]).T
summary_stats_rel = pd.concat([median_values, quartiles], axis=1)
summary_stats_rel.columns = ['Median', '25th Percentile', '75th Percentile']

summary_stats_rel.to_markdown(Path.joinpath(dir_figures,"suppl_table_2.md"))
