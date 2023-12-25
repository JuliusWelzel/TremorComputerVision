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

# Identify the different stages (raw, pca, emd) for mocap
stages = ['raw', 'pca', 'emd']

# Extracting the columns related to 'mocap' for the OMC data and other measurement types
mocap_columns = [col for col in data_new.columns if 'mocap' in col]
other_measurement_columns = [col for col in data_new.columns if 'mocap' not in col and 'freq' in col]

data_new.columns = [col.replace('apple', 'vision') for col in data_new.columns]

# Identify the different stages (raw, pca, emd) for mocap
stages = ['raw', 'pca', 'emd']

# Define the measurement types for comparison
measurement_types = ['Vision', 'Mp', 'Imu']

# Setting up a 2x3 subplot grid (two rows: one for comparing to OMC, one for comparing to IMU)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# First row: comparing Apple and Mp to OMC
for i, stage in enumerate(stages):
    ax = axes[0, i]
    mocap_stage_col = f'mocap_{stage}_freq'

    for measurement_type in measurement_types[:2]:  # Apple and Mp
        other_stage_col = f'{measurement_type.lower()}_{stage}_freq'

        if other_stage_col in data_new.columns:
            # Pairwise removal of NaN values
            paired_data = data_new[[mocap_stage_col, other_stage_col]].dropna()

            if len(paired_data) > 1:
                sns.regplot(x=paired_data[mocap_stage_col], y=paired_data[other_stage_col], ax=ax, scatter_kws={'alpha':0.5}, label=f'{measurement_type}')

                # Calculate the Kendall correlation coefficient
                tau, p_value = stats.kendalltau(paired_data[mocap_stage_col], paired_data[other_stage_col])
                ax.text(0.05, 0.95 - (0.05 * measurement_types.index(measurement_type)), 
                        f'Kendall {measurement_type}: τ={tau:.2f}, p={p_value:.2g}', transform=ax.transAxes)
                

    ax.set_title(f'OMC vs {measurement_types[0]} & {measurement_types[1]} ({stage.upper()})')
    ax.set_xlabel(f'OMC {stage.upper()} Frequency')
    ax.set_ylabel('Frequency')
    ax.legend(loc="lower right")

# Second row: comparing Apple and Mp to IMU
for i, stage in enumerate(stages):
    ax = axes[1, i]
    imu_stage_col = f'imu_{stage}_freq'

    for measurement_type in measurement_types[:2]:  # Apple and Mp
        other_stage_col = f'{measurement_type.lower()}_{stage}_freq'

        if other_stage_col in data_new.columns:
            # Pairwise removal of NaN values
            paired_data = data_new[[imu_stage_col, other_stage_col]].dropna()

            if len(paired_data) > 1:
                sns.regplot(x=paired_data[imu_stage_col], y=paired_data[other_stage_col], ax=ax, scatter_kws={'alpha':0.5}, label=f'{measurement_type}')

                # Calculate the Kendall correlation coefficient
                tau, p_value = stats.kendalltau(paired_data[imu_stage_col], paired_data[other_stage_col])
                ax.text(0.05, 0.95 - (0.05 * measurement_types.index(measurement_type)), 
                        f'Kendall {measurement_type}: τ={tau:.2f}, p={p_value:.2g}', transform=ax.transAxes)

    ax.set_title(f'IMU vs {measurement_types[0]} & {measurement_types[1]} ({stage.upper()})')
    ax.set_xlabel(f'IMU {stage.upper()} Frequency')
    ax.set_ylabel('Frequency')
    ax.legend(loc="lower right")

fig.savefig(Path.joinpath(dir_figures,"figure_6.png"))
plt.show()
