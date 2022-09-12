"""
Variant of x_run.py that performs cross-validation.
One execution of this script will run the model K times and report the mean to Weights & Biases.
"""

from pathlib import Path

import numpy as np
from rich.progress import track
from sklearn.model_selection import KFold
from sklearn.utils import resample

import b_prepare_data
import c_dsea
import c_non_dsea
import d_evaluate
import wandb
import x_config

wandb.init(project="dsea-corn", config=x_config.config, dir=str(Path('build_large/').absolute()))


def run():
    print("Loading data…")
    X, y = b_prepare_data.get_data(dummy=False,
                                   #    to_numpy=True,
                                   nrows=wandb.config.nrows,
                                   )

    # f_test_true = np.bincount(y_test) / wandb.config.num_bins

    n_splits = wandb.config.crossval_n_splits
    kf = KFold(n_splits=n_splits)
    # NOTE: The test size is determined by the number of splits.

    def interim_eval_cb(y_test_pred):
        d_evaluate.evaluate(y_test, y_test_pred)

    metrics_list = []

    for k, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"██████████ Cross-validation iteration {k+1}/{n_splits}")
        # wandb.log({'crossval_iteration': k})
        wandb.log({'crossval_progress': k / n_splits})
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"Training model {k+1}/{n_splits}…")
        y_test_pred = c_dsea.run(X_train, X_test, y_train, interim_eval_cb)

        print(f"Evaluating model {k+1}/{n_splits}…")
        metrics = d_evaluate.evaluate(y_test, y_test_pred, save=False)
        metrics_list.append(metrics)

    # convert [{k: v}, {k: v}] to {k: [v, v]}
    metrics_dict_all = {k: [d[k] for d in metrics_list] for k in metrics_list[0]}
    metrics_dict_mean = {k: np.mean(v) for k, v in metrics_dict_all.items()}
    metrics_dict_std = {k: np.std(v) for k, v in metrics_dict_all.items()}

    # wandb.log(metrics_dict_mean)
    wandb.summary.update({f"{k}_all": v for k, v in metrics_dict_all.items()})
    wandb.summary.update({f"{k}_mean": v for k, v in metrics_dict_mean.items()})

    # If we only logged once per cross-validation iteration, we could use this:
    # https://docs.wandb.ai/guides/track/log#customize-the-summary


if __name__ == '__main__':
    run()
