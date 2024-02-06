from src.config import (dir_figdata, dir_figures, set_style) # import figure directories

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.stats import kruskal
import scikit_posthocs as sp

set_style()

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

#obviously not normal distributed data
sns.histplot(amp_data,x="perc_error_mp_norm_displacement_z")
plt.show()




# Prepare the data for the Kruskal-Wallis test testing the ABSOULTE error.
# define groups to test against each other.
groups = [
    amp_data["error_mp_world_displacement"],
    amp_data["error_mp_world_z_displacement"],
    amp_data["error_mp_norm_displacement"],
    amp_data["error_mp_norm_displacement_z"],
    amp_data["error_apple_displacement"]
]

# Perform the Kruskal-Wallis test
stat, p = kruskal(*groups)
print(f'Kruskal-Wallis Test Statistic: {stat}, p-value: {p}')

# If the Kruskal-Wallis test is significant, proceed with Dunn's post hoc test
if p < 0.051:
    # Combine all groups into a single DataFrame for Dunn's test
    data = pd.melt(amp_data, value_vars=[
        "error_mp_world_displacement",
        "error_mp_world_z_displacement",
        "error_mp_norm_displacement",
        "error_mp_norm_displacement_z",
        "error_apple_displacement"
    ], var_name='Group', value_name='Value')

    # Perform Dunn's test
    p_values_dunn = sp.posthoc_dunn(data, val_col='Value', group_col='Group')
    print(p_values_dunn)
else:
    print("No significant differences were found in the Kruskal-Wallis test.")

#dict of significant comparisons
significant_comparisons={'error_mp_norm_displacement vs error_mp_world_z_displacement': '**',
                         'error_mp_world_z_displacement vs error_apple_displacement':'*',
                            'error_mp_norm_displacement_z vs error_mp_world_z_displacement': '*'}

#set height of the signfincance labels in the plot
manual_heights_new = [150, 170, 160]
manual_heights_dict_new = dict(zip(significant_comparisons.keys(), manual_heights_new))

#list of used methods, usefu for plotting later.
methods = ["error_mp_world_displacement", "error_mp_world_z_displacement",
           "error_mp_norm_displacement", "error_mp_norm_displacement_z",
           "error_apple_displacement"]

# Boxplot of the estimation error relative to OMC in mm
fig, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(data=amp_data[["error_mp_world_displacement", "error_mp_world_z_displacement","error_mp_norm_displacement", "error_mp_norm_displacement_z", "error_apple_displacement"]], ax=ax, palette='colorblind', fliersize=0)
sns.stripplot(data=amp_data[["error_mp_world_displacement", "error_mp_world_z_displacement","error_mp_norm_displacement", "error_mp_norm_displacement_z", "error_apple_displacement"]], ax=ax, color="grey", alpha=.5, size=10)
ax.set_ylabel("Error [mm]")
ax.set_xticklabels(["MP world", "MP world (with z-axis)", "MP norm", "MP norm (with z-axis)", "VI"])

# Adding annotations with asterisks for significance
for comparison, asterisks in significant_comparisons.items():
    method1, method2 = comparison.split(' vs ')
    x1, x2 = methods.index(method1), methods.index(method2)
    y = manual_heights_dict_new[comparison]
    h = 0.5  # Fixed height for the annotation line

    # Drawing lines and asterisk annotations
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='k')
    ax.text((x1+x2)*.5, y+h, asterisks, ha='center', va='bottom', color='k')


fig.savefig(Path.joinpath(dir_figures, "figure_3.png"), dpi=300)
plt.show()





# Prepare the data for the Kruskal-Wallis test testing the RELATIVE error.
# define groups to test against each other.
groups = [
    amp_data["perc_error_mp_world_displacement"],
    amp_data["perc_error_mp_world_z_displacement"],
    amp_data["perc_error_mp_norm_displacement"],
    amp_data["perc_error_mp_norm_displacement_z"],
    amp_data["perc_error_apple_displacement"]
]

# Perform the Kruskal-Wallis test
stat, p = kruskal(*groups)
print(f'Kruskal-Wallis Test Statistic: {stat}, p-value: {p}')

# If the Kruskal-Wallis test is significant, proceed with Dunn's post hoc test
if p < 0.051:
    # Combine all groups into a single DataFrame for Dunn's test
    data = pd.melt(amp_data, value_vars=[
        "perc_error_mp_world_displacement",
        "perc_error_mp_world_z_displacement",
        "perc_error_mp_norm_displacement",
        "perc_error_mp_norm_displacement_z",
        "perc_error_apple_displacement"
    ], var_name='Group', value_name='Value')

    # Perform Dunn's test
    p_values_dunn = sp.posthoc_dunn(data, val_col='Value', group_col='Group')
    print(p_values_dunn)
else:
    print("No significant differences were found in the Kruskal-Wallis test.")

### No significant differences found between the medians of the groups by analysing the relative error.


# Create a figure with three subplots
fig, (ax3, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 20))

colorblind = sns.color_palette("colorblind", 5)
# plot relative error to OMC in % of OMC amplitude vs the OMC amplitude to check for systematic errors.
sns.regplot(x='mocap_displacement', y='error_mp_world_displacement', data=amp_data.dropna(subset=['perc_error_mp_world_displacement']),
            lowess=True, scatter=True, ci=95, label="MP world",color= colorblind[0],ax=ax3)
sns.regplot(x='mocap_displacement', y='error_mp_world_z_displacement', data=amp_data.dropna(subset=['perc_error_mp_world_z_displacement']),
            lowess=True, scatter=True, ci=95, label="MP world (z-axis)",color= colorblind[1],ax=ax3)
sns.regplot(x='mocap_displacement', y='error_mp_norm_displacement', data=amp_data.dropna(subset=['perc_error_mp_norm_displacement']),
            lowess=True, scatter=True, ci=95, label="MP norm",color= colorblind[2],ax=ax3)
sns.regplot(x='mocap_displacement', y='error_mp_norm_displacement_z', data=amp_data.dropna(subset=['perc_error_mp_norm_displacement_z']),
            lowess=True, scatter=True, ci=95, label="MP norm (z-axis)",color= colorblind[3],ax=ax3)
sns.regplot(x='mocap_displacement', y='error_apple_displacement', data=amp_data.dropna(subset=['perc_error_apple_displacement']),
            lowess=True, scatter=True, ci=95, label="VI",color= colorblind[4],ax=ax3)
ax3.legend()
ax3.set_ylabel("Error [mm]")
ax3.set_xlabel("OMC amplitude [mm]")

# plot relative error to OMC in % of OMC amplitude vs the OMC amplitude to check for systematic errors.
sns.regplot(x='mocap_displacement', y='perc_error_mp_world_displacement', data=amp_data.dropna(subset=['perc_error_mp_world_displacement']),
            lowess=True, scatter=True, ci=95, label="MP world",color= colorblind[0],ax=ax1)
sns.regplot(x='mocap_displacement', y='perc_error_mp_world_z_displacement', data=amp_data.dropna(subset=['perc_error_mp_world_z_displacement']),
            lowess=True, scatter=True, ci=95, label="MP world (z-axis)",color= colorblind[1],ax=ax1)
sns.regplot(x='mocap_displacement', y='perc_error_mp_norm_displacement', data=amp_data.dropna(subset=['perc_error_mp_norm_displacement']),
            lowess=True, scatter=True, ci=95, label="MP norm",color= colorblind[2],ax=ax1)
sns.regplot(x='mocap_displacement', y='perc_error_mp_norm_displacement_z', data=amp_data.dropna(subset=['perc_error_mp_norm_displacement_z']),
            lowess=True, scatter=True, ci=95, label= "MP norm (z-axis)",color= colorblind[3],ax=ax1)
sns.regplot(x='mocap_displacement', y='perc_error_apple_displacement', data=amp_data.dropna(subset=['perc_error_apple_displacement']),
            lowess=True, scatter=True, ci=95, label="VI",color= colorblind[4],ax=ax1)

# Correcting the labeling part
ax1.set_ylabel("Error [% of median OMC Amplitude]")  # Changed from ax1.ylabel
ax1.set_xlabel("OMC amplitude [mm]")  # Changed from ax1.xlabel
ax1.legend()
#ax1.set_xscale('log')

sns.boxplot(data=amp_data[["perc_error_mp_world_displacement", "perc_error_mp_world_z_displacement",
                             "perc_error_mp_norm_displacement", "perc_error_mp_norm_displacement_z",
                             "perc_error_apple_displacement"]], ax=ax2, palette='colorblind', fliersize=0)
sns.stripplot(data=amp_data[["perc_error_mp_world_displacement", "perc_error_mp_world_z_displacement",
                               "perc_error_mp_norm_displacement", "perc_error_mp_norm_displacement_z",
                               "perc_error_apple_displacement"]], ax=ax2, color="grey", alpha=.5, size=10)



formatted_p_value = "{:.3f}".format(p)
ax2.text(0.6,0.9, f'Kruskal-Wallis Test, p-value: {formatted_p_value}', transform=ax2.transAxes)

# Labeling axes
ax2.set_ylabel("Error [% of median OMC Amplitude]")
ax2.set_xticklabels(["MP world", "MP world (z-axis)", "MP norm", "MP norm (z-axis)", "VI"])

# Saving the figure with the new manually assigned specific heights for annotations
output_file = Path.joinpath(dir_figures,"figure_4.png")
fig.tight_layout()
fig.savefig(output_file, dpi=300)
plt.show()
