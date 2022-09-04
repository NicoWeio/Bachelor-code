import logging

import numpy as np

from ca_dsea_functions import *
import c_corn
import wandb
from cherenkovdeconvolution import dsea

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def run(X_train, X_test, y_train, interim_eval_callback):
    classifier = c_corn.CornClassifier(
        input_size=X_train.shape[1],
        num_classes=np.bincount(y_train).shape[0],
    )

    def dsea_callback(f, proba, k, alpha, chi2s):
        """
        f: prior
        k: iteration
        alpha: step size
        chi2s: Chi Square distance between iterations
        """
        wandb.log({'f': f, 'k': k, 'alpha': alpha, 'chi2s_iters': chi2s})

        print("▒"*10)
        print(f"Iteration {k} of {wandb.config.num_dsea_iterations}: alpha = {alpha:.3f}, chi2s_iters = {chi2s:.4f}")
        with np.printoptions(precision=3, suppress=True):  # disables scientific notation
            print(f"f = {f}")
        print()

        if proba is not None:  # not set during first iteration
            # TODO: duplicate call during last iteration
            interim_eval_callback(proba)

        if any(f == 0):
            wandb.log({'f_broken': True})
            # raise Exception("f is messed up – aborting…")
            print("WARN: f has entries == 0")
            # TODO: Can we recover from this?

    # alpha=wandb.config.alpha
    alpha = get_adaptive_alpha(X_train, X_test, y_train)

    f_est, probas = dsea(X_test,
                         X_train,
                         y_train,
                         classifier,
                         inspect=dsea_callback,
                         return_contributions=True,
                         K=wandb.config.num_dsea_iterations,
                         fixweighting=wandb.config.fixweighting,
                         alpha=alpha,
                         epsilon=wandb.config.epsilon,
                         )

    return probas
