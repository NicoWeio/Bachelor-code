import time

config = {
    # █ data
    # 'nrows': None, # = all rows
    'nrows': 500_000,  # reasonable subset
    'stratify_test': False,
    'stratify_train': False,
    'num_bins': 10,
    'test_size': 0.2,  # Jan used 0.1

    # █ CORN
    'batch_size': 2**11,  # ✅
    'learning_rate': 0.0025,
    'num_epochs': 10,
    'hidden_units': (120, 240, 120, 12),
    'num_workers': 48,  # no speedup compared to 0

    # █ DSEA
    'num_dsea_iterations': 20, # NOTE: This is only an upper limit; rather tune the epsilon parameter below
    'fixweighting': 'always',  # ✅
    'alpha': 0.5, # ignored when using adaptive step size
    'epsilon': 1e-10,  # TODO
    # 'use_dsea': False, # TODO: respect this
    # ██ TreeDiscretizer
    'J_factor': 10,

    # █ Bootstrap
    'num_bootstrap_iterations': 50,
    # 'num_bootstrap_samples': 500_000,
}


def run_id(config):
    return '_'.join([
        time.strftime('%Y%m%d_%H%M'),
        f"{config['nrows'] or 'all'}samples",
        f"{config['num_epochs']}epochs",
        f"{config['num_dsea_iterations']}dsea",
    ])
