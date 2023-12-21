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
freq_vs_amp=pd.DataFrame()
freq_vs_amp["mp_raw_error"]=np.abs(freq_peaks["mocap_raw_freq"]-freq_peaks["mp_raw_freq"])
freq_vs_amp["apple_raw_error"]=np.abs(freq_peaks["mocap_raw_freq"]-freq_peaks["apple_raw_freq"])
freq_vs_amp["mp_pca_error"]=np.abs(freq_peaks["mocap_pca_freq"]-freq_peaks["mp_pca_freq"])
freq_vs_amp["apple_pca_error"]=np.abs(freq_peaks["mocap_pca_freq"]-freq_peaks["apple_pca_freq"])
freq_vs_amp["mp_emd_error"]=np.abs(freq_peaks["mocap_emd_freq"]-freq_peaks["mp_emd_freq"])
freq_vs_amp["apple_emd_error"]=np.abs(freq_peaks["mocap_emd_freq"]-freq_peaks["apple_emd_freq"])
freq_vs_amp["mocap_displacement"]=np.log(amp_table["mocap_displacement"])


#Plot data for raw, pca, emd for MP and VI. Calculate Kendall correlation.
fig,axes= plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
sns.regplot(x='mocap_displacement', y='mp_raw_error', data=freq_vs_amp, 
            lowess=False, scatter=True, label="MP",color=default_blue,ax=axes[0])
sns.regplot(x='mocap_displacement', y='apple_raw_error', data=freq_vs_amp.dropna(subset=['apple_raw_error']), 
            lowess=False, scatter=True, label="VI",color=default_orange,ax=axes[0])

axes[0].set_ylabel("Frequency estimation error [Hz]")
axes[0].set_xlabel("log(OMC amplitude)")
axes[0].legend(loc="upper right")
axes[0].set_ylim(0,3)
tau_mp, p_value_mp = kendalltau(freq_vs_amp["mocap_displacement"],freq_vs_amp["mp_raw_error"])
axes[0].text(0.05,0.8 , 
                    f'Kendall correlation MP: τ={tau_mp:.2f}, p={p_value_mp:.2g}',
                      transform=axes[0].transAxes)
tau_vi, p_value_vi = kendalltau(freq_vs_amp["apple_raw_error"],freq_vs_amp["mocap_displacement"])
axes[0].text(0.05,0.75, 
                    f'Kendall correlation VI: τ={tau_vi:.2f}, p={p_value_vi:.2g}',
                      transform=axes[0].transAxes)

sns.regplot(x='mocap_displacement', y='mp_pca_error', data=freq_vs_amp, 
            lowess=False, scatter=True, label="MP",color=default_blue,ax=axes[1])
sns.regplot(x='mocap_displacement', y='apple_pca_error', data=freq_vs_amp.dropna(subset=['apple_raw_error']), 
            lowess=False, scatter=True, label="VI",color=default_orange,ax=axes[1])
axes[1].set_ylabel("Frequency estimation error [Hz]")
axes[1].set_xlabel("log(OMC amplitude)")
axes[1].legend(loc="upper right")
axes[1].set_ylim(0,3)
tau_mp, p_value_mp = kendalltau(freq_vs_amp["mocap_displacement"],freq_vs_amp["mp_pca_error"])
axes[1].text(0.05,0.8 , 
                    f'Kendall correlation MP: τ={tau_mp:.2f}, p={p_value_mp:.2g}',
                      transform=axes[1].transAxes)
tau_vi, p_value_vi = kendalltau(freq_vs_amp["mocap_displacement"],freq_vs_amp["apple_pca_error"])
axes[1].text(0.05,0.75 , 
                   f'Kendall correlation VI: τ={tau_vi:.2f}, p={p_value_vi:.2g}',
                     transform=axes[1].transAxes)

sns.regplot(x='mocap_displacement', y='mp_emd_error', data=freq_vs_amp, 
            lowess=False, scatter=True, label="MP",color=default_blue,ax=axes[2])
sns.regplot(x='mocap_displacement', y='apple_emd_error', data=freq_vs_amp.dropna(subset=['apple_raw_error']), 
            lowess=False, scatter=True, label="VI",color=default_orange,ax=axes[2])
axes[2].set_ylabel("Frequency estimation error [Hz]")
axes[2].set_xlabel("log(OMC amplitude")
axes[2].legend(loc="upper right")
axes[2].set_ylim(0,3)
tau_mp, p_value_mp = kendalltau(freq_vs_amp["mocap_displacement"],freq_vs_amp["mp_emd_error"])
axes[2].text(0.05,0.8 , 
                    f'Kendall correlation MP: τ={tau_mp:.2f}, p={p_value_mp:.2g}',
                      transform=axes[2].transAxes)
tau_vi, p_value_vi = kendalltau(freq_vs_amp["mocap_displacement"],freq_vs_amp["apple_emd_error"])
axes[2].text(0.05,0.75 , 
                    f'Kendall correlation VI: τ={tau_vi:.2f}, p={p_value_vi:.2g}',
                      transform=axes[2].transAxes)

plt.legend()
out_path=Path.joinpath(dir_figures,"freq_vs_amp")
plt.savefig(out_path,dpi=300)
plt.show()
