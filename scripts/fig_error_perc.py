from src.config import setup_plt
from src.config import (cfg_colors) # import colors
from src.config import (dir_figdata, dir_figures) # import figure directories

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# setup from config
c_mp = cfg_colors["mediapipe_color"]
c_apple = cfg_colors["apple_color"]
setup_plt(plt)

# load data
peaks_table = pd.read_csv(Path.joinpath(dir_figdata, "hum_displacements_models.csv"))

# boxplot the mp displacements
fig, ax = plt.subplots(figsize=(15, 10))
sns.boxplot(data=peaks_table[["mp_world_displacement", "mp_world_z_displacement","mp_norm_displacement", "mp_norm_displacement_z", "apple_displacement"]], ax=ax, palette='viridis', fliersize=0)
sns.stripplot(data=peaks_table[["mp_world_displacement", "mp_world_z_displacement","mp_norm_displacement", "mp_norm_displacement_z", "apple_displacement"]], ax=ax, color="grey", alpha=.5, size=10)
ax.set_ylabel("Error [mm]")
ax.set_xlabel("Method")
ax.set_title("Median error of methods")

# set font size axis
ax.tick_params(axis='both')
ax.set_xticklabels(["MP world", "MP world (z-axis)", "MP norm", "MP norm (z-axis)", "Apple"])
plt.tight_layout()
plt.show()

# save the plot
fig.savefig(Path.joinpath(dir_figures, "fig_displacements_boxplot.png"), dpi=300)

