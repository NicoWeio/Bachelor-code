import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler

import ba_data_stats
import wandb


def get_data(dummy=True, nrows=None):
    """
    Returns (X, y) ready for training.

    This is fast enough to justify doing it every time.
    """
    df = pd.read_csv('build_large/data.csv', nrows=nrows)

    # discretize the target neutrino energy

    # throw out extreme high and low energy neutrinos
    # TODO: Use under-/overflow bins instead
    lower_limit = 100
    upper_limit = 10**5
    df = df[(df['MCPrimary.energy'] > lower_limit) & (df['MCPrimary.energy'] < upper_limit)]

    # log-scaled Binning
    bins = np.logspace(np.log10(lower_limit), np.log10(upper_limit), wandb.config.num_bins+1)

    # new column with discretized energies
    df['E_discr'] = pd.cut(df['MCPrimary.energy'], bins=bins, labels=range(len(bins)-1))

    X = df.drop(columns=['MCPrimary.energy', 'E_discr'])

    # Standardize features
    # NOTE: Implicit conversion of X to a NumPy array
    # X = StandardScaler().fit_transform(X)

    # Power transform features
    X = PowerTransformer(method='yeo-johnson', standardize=True).fit_transform(X)

    if dummy:
        # Convert categorical variable into dummy/indicator variables
        df_E_dummy = pd.get_dummies(df['E_discr'])
        y = df_E_dummy
    else:
        y = df['E_discr']

    return X, y.to_numpy()


def stratify_data(X, y):
    # We do this manually, since sklearn does not seem to work as intended.
    bincount = np.bincount(y)
    num_bins = len(bincount)
    max_events = np.min(bincount)  # number of events in the smallest bin
    assert max_events > 0

    X_list = []
    y_list = []
    for i in range(num_bins):
        X_list.extend(X[y == i][:max_events])
        y_list.extend(y[y == i][:max_events])

    # Convert to numpy arrays
    X_strat = np.array(X_list)
    y_strat = np.array(y_list)

    # Shuffle the data (!)
    np.random.seed(42)
    perm = np.random.permutation(len(X_strat))
    X_strat = X_strat[perm]
    y_strat = y_strat[perm]

    return X_strat, y_strat


def get_train_test_data():
    X, y = get_data(dummy=False,
                    #    to_numpy=True,
                    nrows=wandb.config.nrows,
                    )
    # y = y.astype(np.int64)  # convert category → int64

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=wandb.config.test_size,
        random_state=42,
    )

    if wandb.config.stratify_train:
        X_train, y_train = stratify_data(X_train, y_train)

    if wandb.config.stratify_test:
        X_test, y_test = stratify_data(X_test, y_test)

    print(ba_data_stats.stats(X_train, X_test, y_train, y_test))

    return X_train, X_test, y_train, y_test
