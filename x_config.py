import time

config_all = {
    # These are the configuration options that are common to all runs and not part of hyperparameter searches.
    #
    # █ data
    # 'nrows': None, # = all rows
    'nrows': 500_000,  # ✅ (Samuel)
    'stratify_test': False,
    'stratify_train': False,
    'num_bins': 10,
    'test_size': 0.1,  # ✅ (Jan / Samuel)
    'underflow_overflow_limits': (10**2.1, 10**5),

    # █ CORN
    'hidden_units': (120, 240, 120, 12),
    'num_workers': 48,  # CPU-dependent; no speedup compared to 0

    # █ DSEA
    'num_dsea_iterations': 20,  # NOTE: This is only an upper limit; rather tune the epsilon parameter below
    'fixweighting': 'always',  # ✅
    'alpha': 0.5,  # ignored when using adaptive step size
    # 'use_dsea': False, # TODO: respect this
}

config_hyperparameters = {
    'batch_size': 4096,  # TODO: maybe 2048 is not actually worse
    'epsilon': 0.01,
    'J': 250,  # for the TreeDiscretizer
    'learning_rate': 0.0004,  # ⚠️
    'num_epochs': 12,  # ✅
}

config_bootstrap = {
    'multi_mode': 'bootstrap',
    'multi_count': 50,  # ≙ num_bootstrap_iterations
}

config_crossval = {
    'multi_mode': 'crossval',
    'multi_count': 10,  # ≙ crossval_n_splits
}

config_bias = {
    'stratify_train': False,
    'stratify_test': True,
}

config_quicktest = {
    'num_epochs': 2,
    'num_dsea_iterations': 2,
    'multi_count': 2,
}

config = config_all | config_hyperparameters | config_bootstrap
# config = config_all | config_hyperparameters | config_crossval

# uncomment this for quick tests ↓
# config |= config_quicktest


def run_id(config):
    return '_'.join([
        time.strftime('%Y%m%d_%H%M'),
        f"{config['nrows'] or 'all'}samples",
        f"{config['num_epochs']}epochs",
        f"{config['num_dsea_iterations']}dsea",
    ])
