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


def get_start_iteration():
    """Returns the number of the first iteration to run."""
    try:
        data_dir = Path('build_large/bootstrap')
        bs_files = data_dir.glob('*.npz')
        iter_nums = [int(f.stem) for f in bs_files]
        # We don't want to repeat the last iteration, so we add 1.
        return max(iter_nums or [0]) + 1
    except Exception as e:
        print(e)
        raise e
        # return 0


def run():
    START_ITER = get_start_iteration()
    if START_ITER > 0:
        print(f"Resuming with iteration {START_ITER}")

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

    bs_results = []

    BS_ITERS = wandb.config.num_bootstrap_iterations

    for bs_iter in range(START_ITER, START_ITER + BS_ITERS):
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

        d_evaluate.save_bootstrap(y_test, y_test_pred, index=(START_ITER + bs_iter)) # TODO: One +1 too much


    d_evaluate.evaluate_bootstrap(bs_results)


if __name__ == '__main__':
    run()
