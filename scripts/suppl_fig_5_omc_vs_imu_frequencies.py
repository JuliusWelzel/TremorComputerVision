from src.config import (cfg_ax_font, cfg_label_font, cfg_title_font, cfg_legend_font) # import font size
from src.config import (cfg_colors) # import colors
from src.config import (dir_figdata, dir_figures) # import figure directories

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path

# Load the data

file_path_new = Path.joinpath(dir_figdata,"human_peaks_table.csv")
data_new = pd.read_csv(file_path_new)

fig, ax = plt.subplots(figsize=(10, 10))
sns.regplot(data=data_new,x="mocap_raw_freq",y="imu_raw_freq",label="OMC vs IMU")
ax.set_xlabel("OMC frequency [Hz]")
ax.set_ylabel("IMU frequency [Hz]")
ax.legend()
tau, p_value = stats.kendalltau(data_new["mocap_raw_freq"], data_new["imu_raw_freq"])
ax.text(0.05, 0.95, f'Kendall correlation: τ={tau:.2f}, p={p_value:.2g}', transform=ax.transAxes)

fig.savefig(Path.joinpath(dir_figures,"suppl_fig_5.png"))

plt.show()
