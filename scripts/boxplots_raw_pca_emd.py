from src.config import (cfg_ax_font, cfg_label_font, cfg_title_font, cfg_legend_font) # import font size
from src.config import (cfg_colors) # import colors
from src.config import (dir_figdata, dir_figures) 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau


# Get the default color cycle and take blue for MP, orange for VI.
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
default_blue = default_colors[0]  # Default blue color
default_orange = default_colors[1]  # Default orange color


#load amplitude and frequency data
path_amplitudes=Path.joinpath(dir_figdata,"displacements.csv")
amp_table=pd.read_csv(path_amplitudes)
path_frequency_peaks=Path.joinpath(dir_figdata,"human_peaks_table.csv")
freq_peaks=pd.read_csv(path_frequency_peaks)


#caluclate frequency estimation error and join with OMC amplitude
data=pd.DataFrame()
data["mp_raw_error"]=np.abs(freq_peaks["mocap_raw_freq"]-freq_peaks["mp_raw_freq"])
data["apple_raw_error"]=np.abs(freq_peaks["mocap_raw_freq"]-freq_peaks["apple_raw_freq"])
data["mp_pca_error"]=np.abs(freq_peaks["mocap_pca_freq"]-freq_peaks["mp_pca_freq"])
data["apple_pca_error"]=np.abs(freq_peaks["mocap_pca_freq"]-freq_peaks["apple_pca_freq"])
data["mp_emd_error"]=np.abs(freq_peaks["mocap_emd_freq"]-freq_peaks["mp_emd_freq"])
data["apple_emd_error"]=np.abs(freq_peaks["mocap_emd_freq"]-freq_peaks["apple_emd_freq"])
#caluclate frequency estimation error and join with OMC amplitude


data=data.melt()

fig, ax = plt.subplots(1, 1, figsize=(15, 12))
sorting_dict = {'mp_raw_error': 0, 'mp_pca_error': 1, 'mp_emd_error': 3,'apple_raw_error': 4, 'apple_pca_error':5, 'apple_emd_error': 6} 
data=data.sort_values(by=['variable'], key=lambda x: x.map(sorting_dict))
sns.boxplot(data=data,x="variable",y="value",palette = ['#ff7f0e','#ff8d28','#ffa85b','#1f77b4','#2b93db','#419ede'],ax=ax,fliersize=0)
ax.set_xticklabels(["MP norm /RAW", "MP norm /PCA", "MP norm / EMD","VI / RAW", "VI / PCA", "VI / EMD"])
sns.stripplot(data=data,x="variable",y="value", ax=ax, color="grey", alpha=.5, size=10)
ax.set_xlabel("")
ax.set_ylabel("Error frequency estimation compared to OMC [Hz]")

out_path=Path.joinpath(dir_figures,"figure_6_error_omc.png")
fig.savefig(out_path)
plt.show()

#caluclate frequency estimation error and join with OMC amplitude
data=pd.DataFrame()
data["mp_raw_error"]=np.abs(freq_peaks["imu_raw_freq"]-freq_peaks["mp_raw_freq"])
data["apple_raw_error"]=np.abs(freq_peaks["imu_raw_freq"]-freq_peaks["apple_raw_freq"])
data["mp_pca_error"]=np.abs(freq_peaks["imu_pca_freq"]-freq_peaks["mp_pca_freq"])
data["apple_pca_error"]=np.abs(freq_peaks["imu_pca_freq"]-freq_peaks["apple_pca_freq"])
data["mp_emd_error"]=np.abs(freq_peaks["imu_emd_freq"]-freq_peaks["mp_emd_freq"])
data["apple_emd_error"]=np.abs(freq_peaks["imu_emd_freq"]-freq_peaks["apple_emd_freq"])
#caluclate frequency estimation error and join with OMC amplitude


data=data.melt()

fig, ax = plt.subplots(1, 1, figsize=(15, 12))
sorting_dict = {'mp_raw_error': 0, 'mp_pca_error': 1, 'mp_emd_error': 3,'apple_raw_error': 4, 'apple_pca_error':5, 'apple_emd_error': 6} 
data=data.sort_values(by=['variable'], key=lambda x: x.map(sorting_dict))
sns.boxplot(data=data,x="variable",y="value",palette = ['#ff7f0e','#ff8d28','#ffa85b','#1f77b4','#2b93db','#419ede'],ax=ax,fliersize=0)
ax.set_xticklabels(["MP norm /RAW", "MP norm /PCA", "MP norm / EMD","VI / RAW", "VI / PCA", "VI / EMD"])
sns.stripplot(data=data,x="variable",y="value", ax=ax, color="grey", alpha=.5, size=10)
ax.set_xlabel("")
ax.set_ylabel("Error frequency estimation compared to IMU [Hz]")
plt.show()

out_path=Path.joinpath(dir_figures,"figure_6_error_imu.png")
fig.savefig(out_path)
plt.show()
