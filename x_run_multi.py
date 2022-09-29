"""
Supports running both bootstrap and cross-validation.
"""

from pathlib import Path

import numpy as np
from rich.progress import track
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import resample

import b_prepare_data
import c_dsea
import c_non_dsea
import d_evaluate
import wandb
import x_config

wandb.init(project="dsea-corn", config=x_config.config, dir=str(Path('build_large/').absolute()))


def get_multi_crossval(X, y):
    n_splits = wandb.config.multi_count
    kf = KFold(n_splits=n_splits)
    # NOTE: The test size is determined by the number of splits.
    if (actual_test_size := 1 / n_splits) != wandb.config.test_size:
        print(f"WARNING: The number of splits ({n_splits}) "
              "does not result in the desired test size "
              f"({wandb.config.test_size} ≠ {actual_test_size}).")

    if wandb.config.stratify_train or wandb.config.stratify_test:
        raise NotImplementedError("Stratification is not implemented for cross-validation.")

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        yield X_train, X_test, y_train, y_test


def get_multi_bootstrap(X, y):
    for j in range(wandb.config.multi_count):
        X_train, X_test, y_train, y_test = b_prepare_data.get_bootstrap(
            X, y,
            n_samples=wandb.config.nrows,  # TODO: make this more transparent
            random_state=j,
        )
        yield X_train, X_test, y_train, y_test


def update_wandb_summary_metrics(metrics_list):
    # convert [{k: v}, {k: v}] to {k: [v, v]}
    metrics_dict_all = {k: [d[k] for d in metrics_list] for k in metrics_list[0]}
    metrics_dict_mean = {k: np.mean(v) for k, v in metrics_dict_all.items()}
    # metrics_dict_std = {k: np.std(v) for k, v in metrics_dict_all.items()}

    # wandb.log(metrics_dict_mean)
    wandb.summary.update({f"{k}_all": v for k, v in metrics_dict_all.items()})
    wandb.summary.update({f"{k}_mean": v for k, v in metrics_dict_mean.items()})


def update_wandb_singleevents(true_labels, predicted_probas):
    """
    Log a random selection of single event probas to wandb.
    The data is logged as [
        # for each true label
        [
            # for each random sample
            [0.1, 0.2, …],  # predicted probas
        ]
    ]
    """
    SAMPLES_PER_CLASS = 10
    sample_indices = []
    for i in range(wandb.config.num_bins):
        possible_indices = np.where(true_labels == i)[0]
        sample_indices_thisbin = np.random.choice(possible_indices, SAMPLES_PER_CLASS, replace=False)
        assert len(sample_indices_thisbin) == SAMPLES_PER_CLASS
        sample_indices.append(sample_indices_thisbin)
    sample_indices = np.array(sample_indices)  # 2D array of shape (num_bins, SAMPLES_PER_CLASS) (I think)

    wandb.summary['single_events'] = predicted_probas[sample_indices].tolist()


def run():
    print("Loading data…")
    X, y = b_prepare_data.get_data(dummy=False,
                                   #    to_numpy=True,
                                   nrows=wandb.config.nrows,
                                   )

    def interim_eval_cb(y_test_pred):
        d_evaluate.evaluate(y_test, y_test_pred)

    multi_bundle = []
    metrics_list = []
    J = wandb.config.multi_count

    multi_getter = {
        'bootstrap': get_multi_bootstrap,
        'crossval': get_multi_crossval,
    }[wandb.config.multi_mode]

    for j, (X_train, X_test, y_train, y_test) in enumerate(multi_getter(X, y)):
        print(f"██████████ Multi iteration {j+1}/{J}")
        # wandb.log({'crossval_iteration': k})

        print(f"Training model {j+1}/{J}…")
        y_test_pred = c_dsea.run(X_train, X_test, y_train, interim_eval_cb)

        print(f"Evaluating model {j+1}/{J}…")
        metrics = d_evaluate.evaluate(y_test, y_test_pred, save=False)
        metrics_list.append(metrics)
        update_wandb_singleevents(y_test, y_test_pred)
        update_wandb_summary_metrics(metrics_list)
        multi_bundle.append((y_test, y_test_pred))
        d_evaluate.evaluate_multi(multi_bundle)

        wandb.log({'multi_progress': (j+1) / J})

    # If we only logged once per cross-validation iteration, we could use this:
    # https://docs.wandb.ai/guides/track/log#customize-the-summary


if __name__ == '__main__':
    run()
