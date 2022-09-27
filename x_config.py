import time

config_all = {
    # █ data
    # 'nrows': None, # = all rows
    'nrows': 500_000,  # ✅ (Samuel)
    'stratify_test': False,
    'stratify_train': False,
    'num_bins': 10,
    'test_size': 0.1,  # ✅ (Jan / Samuel)
    'underflow_overflow_limits': (10**2.1, 10**5),

    # █ CORN
    'batch_size': 2**11,  # ✅
    'learning_rate': 0.0009,
    'num_epochs': 5,
    'hidden_units': (120, 240, 120, 12),
    'num_workers': 48,  # no speedup compared to 0

    # █ DSEA
    'num_dsea_iterations': 20,  # NOTE: This is only an upper limit; rather tune the epsilon parameter below
    'fixweighting': 'always',  # ✅
    'alpha': 0.5,  # ignored when using adaptive step size
    'epsilon': 1e-5,  # TODO
    # 'use_dsea': False, # TODO: respect this
}

config_hyperparameters = {
    'batch_size': 4096,  # TODO: maybe 2048 is not actually worse
    'epsilon': 0.01,
    'J': 100, # for the TreeDiscretizer
    'learning_rate': 0.0009,  # ✓
    'num_epochs': 12,  # ✓
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
