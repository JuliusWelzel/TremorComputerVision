from src.config import (cfg_ax_font, cfg_label_font, cfg_title_font, cfg_legend_font) # import font size
from src.config import (cfg_colors) # import colors
from src.config import (dir_figdata, dir_figures, dir_tables) # import figure directories
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import kruskal
import scikit_posthocs as sp
import numpy as np
from pathlib import Path



# Load the data
file_path_new =Path.joinpath(dir_figdata,'human_peaks_table.csv')
data_new = pd.read_csv(file_path_new)

# Identify the different stages (raw, pca, emd) for mocap
stages = ['raw', 'pca', 'emd']

# Extracting the columns related to 'mocap' for the OMC data and other measurement types
mocap_columns = [col for col in data_new.columns if 'mocap' in col]
other_measurement_columns = [col for col in data_new.columns if 'mocap' not in col and 'freq' in col]

data_new.columns = [col.replace('apple', 'vision') for col in data_new.columns]

# Define the stages and comparisons
stages = ['raw', 'pca', 'emd']
comparisons = [('Vision', 'Mocap'), ('Mp', 'Mocap'), ('Vision', 'Imu'), ('Mp', 'Imu')]

# Prepare a DataFrame to store the results
results_df_new = pd.DataFrame(columns=['Stage', 'Comparison', 'Median Difference', 'IQR 25','IQR 75'])

# Iterate over each stage and each pair of comparisons
for stage in stages:
    for comp in comparisons:
        type_1_col = f'{comp[0].lower()}_{stage}_freq'
        type_2_col = f'{comp[1].lower()}_{stage}_freq'

        if type_1_col in data_new.columns and type_2_col in data_new.columns:
            # Calculate absolute differences
            differences = abs(data_new[type_1_col] - data_new[type_2_col])
            median_diff = differences.median()
            iqr25 = differences.quantile(0.25)
            iqr75 =differences.quantile(0.75) 

            # Append results to the dataframe
            results_df_new = results_df_new._append({
                'Stage': stage.upper(),
                'Comparison': f'{comp[0]} vs {comp[1]}',
                'Median Difference': median_diff,
                'IQR 25': iqr25,
                'IQR 75': iqr75
            }, ignore_index=True)

print(results_df_new)
results_df_new.to_markdown(dir_tables.joinpath("table_1.md"))


## statistics, compare along the frameworks (VI,MP) and along the Pipelines (RAW, PCA, EMD) for both OMC and IMU error

error_table=pd.DataFrame()
error_table["mp_mocap_raw"]=np.abs(data_new["mocap_raw_freq"]-data_new["mp_raw_freq"])
error_table["vi_mocap_raw"]=np.abs(data_new["mocap_raw_freq"]-data_new["vision_raw_freq"])
error_table["mp_mocap_pca"]=np.abs(data_new["mocap_pca_freq"]-data_new["mp_pca_freq"])
error_table["vi_mocap_pca"]=np.abs(data_new["mocap_pca_freq"]-data_new["vision_pca_freq"])
error_table["mp_mocap_emd"]=np.abs(data_new["mocap_emd_freq"]-data_new["mp_emd_freq"])
error_table["vi_mocap_emd"]=np.abs(data_new["mocap_emd_freq"]-data_new["vision_emd_freq"])
error_table["mp_imu_raw"]=np.abs(data_new["imu_raw_freq"]-data_new["mp_raw_freq"])
error_table["vi_imu_raw"]=np.abs(data_new["imu_raw_freq"]-data_new["vision_raw_freq"])
error_table["mp_imu_pca"]=np.abs(data_new["imu_pca_freq"]-data_new["mp_pca_freq"])
error_table["vi_imu_pca"]=np.abs(data_new["imu_pca_freq"]-data_new["vision_pca_freq"])
error_table["mp_imu_emd"]=np.abs(data_new["imu_emd_freq"]-data_new["mp_emd_freq"])
error_table["vi_imu_emd"]=np.abs(data_new["imu_emd_freq"]-data_new["vision_emd_freq"])

stat,p_value=stats.mannwhitneyu(error_table["mp_mocap_raw"],error_table["vi_mocap_raw"])
print(p_value)

stat,p_value=stats.mannwhitneyu(error_table["mp_mocap_pca"],error_table["vi_mocap_pca"])
print(p_value)

stat,p_value=stats.mannwhitneyu(error_table["mp_mocap_emd"],error_table["vi_mocap_emd"])
print(p_value)

stat,p_value=stats.mannwhitneyu(error_table["mp_imu_raw"],error_table["vi_imu_raw"])
print(p_value)

stat,p_value=stats.mannwhitneyu(error_table["mp_imu_pca"],error_table["vi_imu_pca"])
print(p_value)

stat,p_value=stats.mannwhitneyu(error_table["mp_imu_emd"],error_table["vi_imu_emd"])
print(p_value)




#Test MP mocap across pipelines.
# define groups to test against each other.
groups = [
    error_table["mp_mocap_raw"],
    error_table["mp_mocap_pca"],
    error_table["mp_mocap_emd"]
]

# Perform the Kruskal-Wallis test
stat, p = kruskal(*groups)
print(f'Kruskal-Wallis Test Statistic: {stat}, p-value: {p}')

# If the Kruskal-Wallis test is significant, proceed with Dunn's post hoc test
if p < 0.05:
    # Combine all groups into a single DataFrame for Dunn's test
    data = pd.melt(error_table, value_vars=[
        "mp_mocap_raw",
        "mp_mocap_pca",
        "mp_mocap_emd"
    ], var_name='Group', value_name='Value')

    # Perform Dunn's test
    p_values_dunn = sp.posthoc_dunn(data, val_col='Value', group_col='Group')
    print(p_values_dunn)
else:
    print("No significant differences were found in the Kruskal-Wallis test.")




#Test MP imu across pipelines.
# define groups to test against each other.
groups = [
    error_table["mp_imu_raw"],
    error_table["mp_imu_pca"],
    error_table["mp_imu_emd"]
]

# Perform the Kruskal-Wallis test
stat, p = kruskal(*groups)
print(f'Kruskal-Wallis Test Statistic: {stat}, p-value: {p}')

# If the Kruskal-Wallis test is significant, proceed with Dunn's post hoc test
if p < 0.05:
    # Combine all groups into a single DataFrame for Dunn's test
    data = pd.melt(error_table, value_vars=[
        "mp_imu_raw",
        "mp_imu_pca",
        "mp_imu_emd"
    ], var_name='Group', value_name='Value')

    # Perform Dunn's test
    p_values_dunn = sp.posthoc_dunn(data, val_col='Value', group_col='Group')
    print(p_values_dunn)
else:
    print("No significant differences were found in the Kruskal-Wallis test.")




#Test VI mocap across pipelines.
# define groups to test against each other.
groups = [
    error_table["vision_mocap_raw"],
    error_table["vision_mocap_pca"],
    error_table["vision_mocap_emd"]
]

# Perform the Kruskal-Wallis test
stat, p = kruskal(*groups)
print(f'Kruskal-Wallis Test Statistic: {stat}, p-value: {p}')

# If the Kruskal-Wallis test is significant, proceed with Dunn's post hoc test
if p < 0.05:
    # Combine all groups into a single DataFrame for Dunn's test
    data = pd.melt(error_table, value_vars=[
        "vision_mocap_raw",
        "visiom_mocap_pca",
        "vision_mocap_emd"
    ], var_name='Group', value_name='Value')

    # Perform Dunn's test
    p_values_dunn = sp.posthoc_dunn(data, val_col='Value', group_col='Group')
    print(p_values_dunn)
else:
    print("No significant differences were found in the Kruskal-Wallis test.")



#Test VI mocap across pipelines.
# define groups to test against each other.
groups = [
    error_table["vision_imu_raw"],
    error_table["vision_imu_pca"],
    error_table["vision_imu_emd"]
]

# Perform the Kruskal-Wallis test
stat, p = kruskal(*groups)
print(f'Kruskal-Wallis Test Statistic: {stat}, p-value: {p}')

# If the Kruskal-Wallis test is significant, proceed with Dunn's post hoc test
if p < 0.05:
    # Combine all groups into a single DataFrame for Dunn's test
    data = pd.melt(error_table, value_vars=[
        "vision_imu_raw",
        "visiom_imu_pca",
        "vision_imu_emd"
    ], var_name='Group', value_name='Value')

    # Perform Dunn's test
    p_values_dunn = sp.posthoc_dunn(data, val_col='Value', group_col='Group')
    print(p_values_dunn)
else:
    print("No significant differences were found in the Kruskal-Wallis test.")



# None of the comparisons above was signicant anyway, therefore the multiple testing correction is not relevant anymore.
