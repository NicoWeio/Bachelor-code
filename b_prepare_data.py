import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler

import ba_data_stats
import wandb


def get_bin_edges(df, limits, num_bins_between_limits, overflow=True):
    """
    Returns the log-scaled bin edges (including both leftmost and rightmost) based on a given DataFrame.
    Under-/overflow bins are added if they don't conflict with the limits.
    """
    lower_limit, upper_limit = limits

    # TODO: use geomspace instead of logspace
    # log_bin_edges = np.logspace(np.log10(lower_limit), np.log10(upper_limit), num_bins_between_limits+1)
    log_bin_edges = np.geomspace(lower_limit, upper_limit, num_bins_between_limits+1)

    if not overflow:
        return log_bin_edges

    the_min = min(df['MCPrimary.energy'])
    the_max = max(df['MCPrimary.energy'])

    if the_min >= lower_limit:
        print(
            f"Warning: The minimum energy in the dataset is {the_min:.2e} GeV, which is larger than (or equal to) the lower limit {lower_limit:.2e} GeV.")
        print("No underflow bin will be created.")
    if the_max <= upper_limit:
        print(
            f"Warning: The maximum energy in the dataset is {the_max:.2e} GeV, which is smaller than (or equal to) the upper limit {upper_limit:.2e} GeV.")
        print("No overflow bin will be created.")

    # This represents bin edges, including the leftmost and rightmost ones.
    bin_edges = np.concatenate([
        # [1],  # [0] might cause problems with the log scale
        [the_min] if the_min < lower_limit else [],
        log_bin_edges,
        [the_max] if the_max > upper_limit else [],
        # [np.infty],
    ])

    # FIXME: Underflow bin may or may not exist depending on the custom limits.
    # assert len(bin_edges) == NUM_BINS + 1, f"Expected {NUM_BINS+1} bin edges, got {len(bin_edges)}."

    return bin_edges


def get_data(dummy=True, nrows=None):
    """
    Returns (X, y) ready for training.

    This is fast enough to justify doing it every time.
    """
    df = pd.read_csv('build_large/data.csv', nrows=nrows)

    # █ Discretize the target neutrino energy
    # ↓ variant with under-/overflow bins
    # bin_edges = get_bin_edges(df, (LOWER_LIMIT, UPPER_LIMIT), (wandb.config.num_bins-1))
    bin_edges = get_bin_edges(df, wandb.config.underflow_overflow_limits, num_bins_between_limits=(wandb.config.num_bins-2))
    # Ensure that we always have num_bins total bins, including under-/overflow bins.
    # We need to change num_bins_between_limits if the under-/overflow bins are omitted.
    assert len(bin_edges) == wandb.config.num_bins + 1
    wandb.summary['bin_edges'] = bin_edges

    # ↓ variant without under-/overflow bins
    # bin_edges = get_bin_edges(df,
    #                           (int(min(df['MCPrimary.energy']) - 1), int(max(df['MCPrimary.energy']) + 1)),
    #                           wandb.config.num_bins,
    #                           overflow=False,
    #                           )

    # new column with discretized energies
    df['E_discr'] = pd.cut(df['MCPrimary.energy'], bins=bin_edges, labels=range(len(bin_edges)-1), include_lowest=True)

    X = df.drop(columns=['MCPrimary.energy', 'E_discr'])

    # █ Standardize features
    # NOTE: Implicit conversion of X to a NumPy array
    # X = StandardScaler().fit_transform(X)

    # █ Power transform features
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


def get_bootstrap(X, y, n_samples, random_state=42):
    from numpy.random import default_rng
    rng = default_rng(seed=random_state)

    # get a np array of random indices with replacement
    # (i.e. each index is drawn with equal probability)
    # indices_bs = np.random.randint(0, len(X), size=n_samples, random_state=random_state)
    indices_bs = rng.integers(0, len(X), size=n_samples)
    # indices_bs = np.random.choice(len(X), size=n_samples, replace=True, random_state=random_state)

    indices_not_bs = np.array(list(set(range(len(X))) - set(indices_bs)))

    # train / bootstrapped data
    X_train = X[indices_bs].copy()
    y_train = y[indices_bs].copy()
    if wandb.config.stratify_train:
        raise NotImplementedError("Stratification of bootstrap training data is not implemented.")

    # test / out-of-bag data
    X_test = X[indices_not_bs].copy()
    y_test = y[indices_not_bs].copy()
    if wandb.config.stratify_test:
        # raise NotImplementedError("Stratification of bootstrap test data is not implemented.")
        X_test, y_test = stratify_data(X_test, y_test)

    print(ba_data_stats.stats(X_train, X_test, y_train, y_test))

    return X_train, X_test, y_train, y_test
