import time

config = {
    # █ data
    # 'nrows': None, # = all rows
    'nrows': 200000,  # reasonable subset
    'num_bins': 10,
    # █ CORN
    'batch_size': 2**11,  # ✅
    'learning_rate': 0.0001159,
    'num_epochs': 5,
    'hidden_units': (120, 240, 120, 12),
    'num_workers': 48,  # no speedup compared to 0
    # █ DSEA
    'num_dsea_iterations': 5,
    'fixweighting': True,
}


def run_id(config):
    return '_'.join([
        time.strftime('%Y%m%d_%H%M'),
        f"{config['nrows'] or 'all'}samples",
        f"{config['num_epochs']}epochs",
        f"{config['num_dsea_iterations']}dsea",
    ])
