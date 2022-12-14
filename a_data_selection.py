import pandas as pd
from pathlib import Path

feature_list = Path('a_data_selection_features.csv').read_text().strip().split('\n')

df = pd.read_csv('/net/big-tank/POOL/users/lkardum/new_mc_binning.csv')
old_size = df.memory_usage(index=True).sum()

# drop critical rows
df.drop(
    [1007671, 1426233, 2304853, 2715790, 3674355, 3741687, 4178063, 4969266, 5038333, 5334552, 5589516, 5863719,
     5978972, 7006367, 7281704, 7509650, 8380383, 8758113, 9043798, 10280382, 11179530, 11184928, 11332586, 11797767, 12253944],
    axis=0,
    inplace=True,
)

# substitute NaNs with extreme value
df.fillna(value=-100000, inplace=True)

df = df[feature_list]

new_size = df.memory_usage(index=True).sum()

print(f"Size reduced from {old_size} to {new_size} ({(new_size/old_size):.0%} of original)")

df.to_csv('build_large/data.csv', index=False)
