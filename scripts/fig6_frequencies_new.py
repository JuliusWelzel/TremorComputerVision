from src.config import (cfg_ax_font, cfg_label_font, cfg_title_font, cfg_legend_font) # import font size
from src.config import (cfg_colors) # import colors
from src.config import (dir_figdata, dir_figures) 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
import numpy as np

# Get the default color cycle
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
default_blue = default_colors[0]  # Default blue color
default_orange = default_colors[1]  # Default orange color

# Load the data
file_path_new = 'figdata\human_peaks_table.csv'
data_new = pd.read_csv(file_path_new)
data_new.columns = [col.replace('apple', 'vision') for col in data_new.columns]

idx_apple_omc=abs(data_new["vision_raw_freq"]-data_new["mocap_raw_freq"])>1.0
idx_mp_omc=abs(data_new["mp_raw_freq"]-data_new["mocap_raw_freq"])>1.0
idx_apple_imu=abs(data_new["vision_raw_freq"]-data_new["imu_raw_freq"])>1.0
idx_mp_imu=abs(data_new["mp_raw_freq"]-data_new["imu_raw_freq"])>1.0

data_new["mocap_raw_freq"][~data_new["mocap_raw_freq"].index.isin(idx_mp_omc)]


fig, axes = plt.subplots(2, 2, figsize=(18, 12))

sns.regplot(data_new[~idx_mp_omc],x="mocap_raw_freq",y="mp_raw_freq",ax=axes[0,0],label="MP norm vs OMC",color=default_orange)
tau, p_value = stats.kendalltau(data_new["mocap_raw_freq"][~idx_mp_omc], 
                                data_new["mp_raw_freq"][~idx_mp_omc])
axes[0,0].text(0.05, 0.90 , 
                        f'Kendall correlation: τ={tau:.2f}, p={p_value:.2g}', transform=axes[0,0].transAxes)
axes[0,0].set_xlabel("OMC frequency [Hz]")
axes[0,0].set_ylabel("MP norm frequency [Hz]")
#axes[0,0].legend(loc="upper right")
axes[0,0].scatter(data_new["mocap_raw_freq"][idx_mp_omc],data_new["mp_raw_freq"][idx_mp_omc],color="red",marker="x")
axes[0,0].set_xlim(3,8)
axes[0,0].set_ylim(3,8)
axes[0,0].text(-0.1,1.1,"A",transform=axes[0,0].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')


sns.regplot(data_new[~idx_mp_imu],x="imu_raw_freq",y="mp_raw_freq",ax=axes[1,0],label="MP norm vs IMU",color=default_orange)
tau, p_value = stats.kendalltau(data_new["imu_raw_freq"][~idx_mp_imu], 
                                data_new["mp_raw_freq"][~idx_mp_imu])
axes[1,0].text(0.05, 0.90, f'Kendall correlation: τ={tau:.2f}, p={p_value:.2g}', transform=axes[1,0].transAxes)
axes[1,0].set_xlabel("IMU frequency [Hz]")
axes[1,0].set_ylabel("MP norm frequency [Hz]")
#axes[1,0].legend(loc="upper right")
axes[1,0].scatter(data_new["imu_raw_freq"][idx_mp_imu],data_new["mp_raw_freq"][idx_mp_imu],color="red",marker="x")
axes[1,0].set_xlim(3,8)
axes[1,0].set_ylim(3,8)
axes[1,0].text(-0.1,1.1,"B",transform=axes[1,0].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')


sns.regplot(data_new[~idx_apple_omc],x="mocap_raw_freq",y="vision_raw_freq",ax=axes[0,1],label="VI vs OMC",color=default_blue)
tau, p_value = stats.kendalltau(data_new["mocap_raw_freq"][~idx_apple_omc], 
                                data_new["vision_raw_freq"][~idx_apple_omc])
axes[0,1].text(0.05, 0.90 , 
                        f'Kendall correlation: τ={tau:.2f}, p={p_value:.2g}', transform=axes[0,1].transAxes)
axes[0,1].set_xlabel("OMC frequency [Hz]")
axes[0,1].set_ylabel("VI frequency [Hz]")
#axes[0,1].legend(loc="upper right")
axes[0,1].scatter(data_new["mocap_raw_freq"][idx_apple_omc],data_new["vision_raw_freq"][idx_apple_omc],color="red",marker="x")
axes[0,1].set_xlim(3,8)
axes[0,1].set_ylim(3,8)
axes[0,1].text(-0.1,1.1,"C",transform=axes[0,1].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')


sns.regplot(data_new[~idx_apple_imu],x="imu_raw_freq",y="vision_raw_freq",ax=axes[1,1],label="VI vs IMU",color=default_blue)
tau, p_value = stats.kendalltau(data_new["imu_raw_freq"][~idx_apple_imu], 
                                data_new["vision_raw_freq"][~idx_apple_imu])
axes[1,1].text(0.05, 0.90 , 
                    f'Kendall correlation: τ={tau:.2f}, p={p_value:.2g}', transform=axes[1,1].transAxes)
axes[1,1].set_xlabel("IMU frequency [Hz]")
axes[1,1].set_ylabel("VI frequency [Hz]")
#axes[1,1].legend(loc="upper right")
axes[1,1].scatter(data_new["imu_raw_freq"][idx_apple_imu],data_new["vision_raw_freq"][idx_apple_imu],color="red",marker="x")
axes[1,1].set_xlim(3,8)
axes[1,1].set_ylim(3,8)
axes[1,1].text(-0.1,1.1,"D",transform=axes[1,1].transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

fig.savefig(Path.joinpath(dir_figures,"frequencies_raw.png"),dpi=300)
plt.show()

# Selecting the required columns for median and IQR calculations

data_new["error_"]


# Table and Statistics of frequency estimation

estimation_errors=pd.DataFrame()
estimation_errors["MP vs OMC"]=np.abs(data_new["mocap_raw_freq"]-data_new["mp_raw_freq"])
estimation_errors["MP vs IMU"]=np.abs(data_new["imu_raw_freq"]-data_new["mp_raw_freq"])
estimation_errors["VI vs OMC"]=np.abs(data_new["mocap_raw_freq"]-data_new["vision_raw_freq"])
estimation_errors["VI vs IMU"]=np.abs(data_new["imu_raw_freq"]-data_new["vision_raw_freq"])

# Calculate the median and IQR values for the selected columns
min_value=estimation_errors.min()
max_value=estimation_errors.max()
median = estimation_errors.median()
iqr_25 = estimation_errors.quantile(0.25)
iqr_75 = estimation_errors.quantile(0.75)

# Creating a DataFrame to display the results in a nice format
stats_df = pd.DataFrame({
    'Median': median,
    "Minimum": min_value,
    'IQR 25': iqr_25,
    'IQR 75': iqr_75,
    "Maximum": max_value
})

# Calculating the IQR range (from 25th to 75th percentile)
#stats_df['IQR Range'] = stats_df['IQR 75%'] - stats_df['IQR 25%']
stats_df.to_markdown("tables/table_frequencies_raw.md")


from scipy.stats import mannwhitneyu,shapiro

shapiro(estimation_errors["MP vs OMC"])

res1=mannwhitneyu(estimation_errors["MP vs OMC"],estimation_errors["VI vs OMC"])
res2=mannwhitneyu(estimation_errors["MP vs IMU"],estimation_errors["VI vs IMU"])

print(res1)
print(res2)


data_new = pd.read_csv(file_path_new)

fig, axes = plt.subplots(1,1, figsize=(12, 12))

sns.regplot(data_new,x="mocap_raw_freq",y="imu_raw_freq",ax=axes,label="OMC vs IMU",color=default_blue)
tau, p_value = stats.kendalltau(data_new["mocap_raw_freq"], data_new["imu_raw_freq"])
axes.text(0.05, 0.95 , 
                        f'Kendall correlation: τ={tau:.2f}, p={p_value:.2g}', transform=axes.transAxes)
axes.set_xlabel("OMC frequency [Hz]")
axes.set_ylabel("IMU frequency [Hz]")
axes.legend(loc="upper right")
plt.show()

fig.savefig("figures/IMUvsOMC.png")

np.quantile(np.abs(data_new["mocap_raw_freq"]-data_new["imu_raw_freq"]),0.25)
np.quantile(np.abs(data_new["mocap_raw_freq"]-data_new["imu_raw_freq"]),0.75)
np.median(np.abs(data_new["mocap_raw_freq"]-data_new["imu_raw_freq"]))


