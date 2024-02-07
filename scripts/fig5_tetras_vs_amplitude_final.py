from src.config import (cfg_ax_font, cfg_label_font, cfg_title_font, cfg_legend_font, set_style) # import font size
from src.config import (cfg_colors) # import colors
from src.config import (dir_figdata, dir_figures) # import figure directories
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
import numpy as np
from scipy.stats import zscore
from scipy.stats import kendalltau

# set plotting configs
set_style()
colors_cv_models = cfg_colors["cv_model_colors"]
c_map_cv_models = cfg_colors["cv_model_colors"].values()

# Get the default color cycle
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
default_blue = default_colors[0]  # Default blue color
default_orange = default_colors[1]  # Default orange color

peaks_table=pd.read_csv(Path.joinpath(dir_figdata,"hum_displacements_models.csv"))
tetras=pd.read_csv(Path.joinpath(dir_figdata,"hum_patient_info.csv"),delimiter=";")["TETRAS"]

peaks_table["tetras"]=tetras

bins = [0, 5, 30, 100,1000]
group_names = [1, 2, 3, 4]

peaks_table['tetras_mocap'] = pd.cut(peaks_table['mocap_displacement'], bins=bins, labels=group_names)


peaks_table["mocap_displacement"]=(np.log(peaks_table["mocap_displacement"]))
peaks_table["mp_world_displacement"]=(np.log(peaks_table["mp_world_displacement"]))
peaks_table["mp_world_z_displacement"]=(np.log(peaks_table["mp_world_z_displacement"]))
peaks_table["mp_norm_displacement"]=(np.log(peaks_table["mp_norm_displacement"]))
peaks_table["mp_norm_displacement_z"]=(np.log(peaks_table["mp_norm_displacement_z"]))
peaks_table["apple_displacement"]=(np.log(peaks_table["apple_displacement"]))

viridis = sns.color_palette("colorblind", 5)
# Create a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 20))
# Plot each regplot in a separate subplot
sns.regplot(data=peaks_table, x="tetras", y="mocap_displacement", ax=axes[0, 0],label="log(OMC Amplitude)",color="black")
sns.regplot(data=peaks_table, x="tetras", y="mp_world_displacement", ax=axes[0, 1],label="log(MP world Amplitude)",color=colors_cv_models["MPworld"])
sns.regplot(data=peaks_table, x="tetras", y="mp_world_z_displacement", ax=axes[0, 2],label="log(MP world (with z-axis) Amplitude)",color=colors_cv_models["MPworld_z"])
sns.regplot(data=peaks_table, x="tetras", y="mp_norm_displacement", ax=axes[1, 0],label="log(MP norm Amplitude)",color=colors_cv_models["MPnorm"])
sns.regplot(data=peaks_table, x="tetras", y="mp_norm_displacement_z", ax=axes[1, 1],label="log(MP norm (with z-axis) Amplitude)",color=colors_cv_models["MPnorm_z"])
sns.regplot(data=peaks_table, x="tetras", y="apple_displacement", ax=axes[1, 2],label="log(VI Amplitude)",color=colors_cv_models["Apple_VI"])
axes[0, 0].set_ylabel("log(OMC Amplitude)")
axes[0, 1].set_ylabel("log(MP world Amplitude)")
axes[0, 2].set_ylabel("log(MP world (with z-axis) Amplitude)")
axes[1, 0].set_ylabel("log(MP norm Amplitude)")
axes[1, 1].set_ylabel("log(MP norm (with z-axis) Amplitude)")
axes[1, 2].set_ylabel("log(VI Amplitude)")

tau, p_value = kendalltau(peaks_table["tetras"],peaks_table["mocap_displacement"])
axes[0, 0].text(0.05, 0.95 , 
                    f'Kendall correlation:\n τ={tau:.2f}, p={p_value:.2g}',
                      transform=axes[0,0].transAxes)
tau, p_value = kendalltau(peaks_table["tetras"],peaks_table["mp_world_displacement"])
axes[0, 1].text(0.05, 0.95 , 
                    f'Kendall correlation:\n τ={tau:.2f}, p={p_value:.2g}',
                      transform=axes[0,1].transAxes)
tau, p_value = kendalltau(peaks_table["tetras"],peaks_table["mp_world_z_displacement"])
axes[0, 2].text(0.05, 0.95 , 
                    f'Kendall correlation:\n τ={tau:.2f}, p={p_value:.2g}',
                      transform=axes[0,2].transAxes)
tau, p_value = kendalltau(peaks_table["tetras"],peaks_table["mp_norm_displacement"])
axes[1, 0].text(0.05, 0.95 , 
                    f'Kendall correlation:\n τ={tau:.2f}, p={p_value:.2g}',
                      transform=axes[1,0].transAxes)
tau, p_value = kendalltau(peaks_table["tetras"],peaks_table["mp_norm_displacement_z"])
axes[1, 1].text(0.05, 0.95 , 
                    f'Kendall correlation:\n τ={tau:.2f}, p={p_value:.2g}',
                      transform=axes[1,1].transAxes)
tau, p_value = kendalltau(peaks_table["tetras"],peaks_table["apple_displacement"])
axes[1, 2].text(0.05, 0.95 , 
                    f'Kendall correlation:\n τ={tau:.2f} p={p_value:.2g}',
                      transform=axes[1,2].transAxes)

# Adjust layout for better visualization
fig.savefig(Path.joinpath(dir_figures,"figure_5"))
plt.tight_layout()
plt.show()

