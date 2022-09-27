"""
Uploads a bootstrap run to wandb after it has finished, using the .npz files.
"""
from pathlib import Path

import d_evaluate
import wandb
import b_prepare_data
import x_config  # NOTE: Does not necessarily represent the config used for the run

wandb.init(project="dsea-corn", config=x_config.config, dir=str(Path('build_large/').absolute()),
           job_type="upload_bootstrap")

# ↓ logs the bin edges
b_prepare_data.get_data(
    dummy=False,
    nrows=max(1_000_000, wandb.config.nrows),
)

bs_bundle = d_evaluate.load_bootstrap_bundle()
d_evaluate.evaluate_bootstrap(bs_bundle)
