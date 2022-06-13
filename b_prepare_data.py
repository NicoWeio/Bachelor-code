# Draft for externalizing data preparation.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_data(dummy=True, to_numpy=True, nrows=None):
    """Returns (X, y) ready for training."""
    df = pd.read_csv('build_large/data.csv', nrows=nrows)

    # discretize the target neutrino energy

    # Variables: Label
    # drop out all events with energies outside the range
    lower_limit = 100
    upper_limit = 10**5

    # Variables: NN
    num_bins = 10  # number of bins (energy classes), output_shape

    # throw out extreme high and low energy neutrinos
    df = df[(df['MCPrimary.energy'] < upper_limit) & (df['MCPrimary.energy'] > lower_limit)]

    # log-scaled Binning
    bins = np.logspace(np.log10(lower_limit), np.log10(upper_limit), num_bins+1)

    # new column with discretized energies
    df['E_discr'] = pd.cut(df['MCPrimary.energy'], bins=bins, labels=range(len(bins)-1))

    X = df.drop(columns=['MCPrimary.energy', 'E_discr'])

    if dummy:
        # Convert categorical variable into dummy/indicator variables
        df_E_dummy = pd.get_dummies(df['E_discr'])
        y = df_E_dummy
    else:
        y = df['E_discr']


    if to_numpy:
        return X.to_numpy(), y.to_numpy()
    else:
        return X, y
