import pandas as pd
from pathlib import Path
from src.config import (set_style) # import font size
from src.config import (cfg_colors) # import colors
from src.config import (dir_figdata, dir_figures, dir_tables) # import figure directories

# set plotting configs
set_style()

#1 read data
data=pd.read_csv(Path.joinpath(dir_figdata,"hum_patient_info.csv"),delimiter=";")

#2 Age
age_mean = data['subject_number'].mean()
age_std = data['subject_number'].std()

# 3. Tremor type (rest/hold)
tremor_counts = data['tremor_condition'].value_counts()

# 4. TETRAS Tremor of upper limb (0-5)
tETRAS_mean = data['TETRAS'].mean()
tETRAS_std = data['TETRAS'].std()

# 5. Diagnosis
diagnosis_counts = data['diagnosis'].value_counts()

# Preparing the summary table
summary_table = pd.DataFrame({
    "Category": ["Age (years)", "Tremor type (rest/hold)", "TETRAS Tremor of upper limb (0-5)", "Diagnosis"],
    "Statistics": [f"{age_mean:.1f} ({age_std:.1f})", 
                   f"{tremor_counts['rest']}/{tremor_counts['hold']}", 
                   f"{tETRAS_mean:.3f} ({tETRAS_std:.2f})", 
                   diagnosis_counts.to_string().replace('\n', ', ')]
})

# Displaying the summary table
summary_table.to_markdown(dir_tables.joinpath("table_2.md"),index=False)
