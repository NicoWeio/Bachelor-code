from pathlib import Path

from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_validate
from sklearn.utils import resample

import b_prepare_data
import c_dsea
import d_evaluate
import wandb
import x_config

# TODO: Raises a warning…
wandb.init(project="dsea-corn", config=x_config.config, dir=Path('build_large/').absolute())


def run():
    print("Loading data…")
    # X_train, X_test, y_train, y_test = b_prepare_data.get_train_test_data()
    X, y = b_prepare_data.get_data(dummy=False,
                                   # to_numpy=True,
                                   #    nrows=wandb.config.nrows,
                                   # Since we train on a bootstrapped sample, let's load some more.
                                   # This should lead to less occasions of multiple draws, therefore providing a more diverse training set.
                                   nrows=max(1_000_000, wandb.config.nrows),
                                    #    nrows=None,
                                   )

    # NOTE: Samuel behält die Trainings-Daten über Bootstrap-Iterationen bei!?

    bs_results = []

    BS_ITERS = wandb.config.num_bootstrap_iterations
    for bs_iter in range(1, BS_ITERS+1):
        print(f"██████████ Bootstrap iteration {bs_iter}/{BS_ITERS}")

        X_train, X_test, y_train, y_test = b_prepare_data.get_bootstrap(
            X, y,
            n_samples=wandb.config.nrows,  # TODO: make this more transparent
            random_state=bs_iter,
        )

        def interim_eval_cb(y_test_pred):
            # d_evaluate.evaluate(y_test, y_test_pred)
            return

        print("Training model…")
        y_test_pred = c_dsea.run(X_train, X_test, y_train, interim_eval_cb)

        # print("Evaluating model…")
        # d_evaluate.evaluate_bootstrap(y_test, y_test_pred, save=False)

        bs_results.append((y_test, y_test_pred))

    d_evaluate.evaluate_bootstrap(bs_results, save=True)


if __name__ == '__main__':
    run()
