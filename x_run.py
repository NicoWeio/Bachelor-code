from pathlib import Path

from rich.progress import track
from sklearn.utils import resample

import b_prepare_data
import c_dsea
import c_non_dsea
import d_evaluate
import wandb
import x_config

# TODO: Raises a warning…
wandb.init(project="dsea-corn", config=x_config.config, dir=Path('build_large/').absolute())


def run():
    print("Loading data…")
    X_train, X_test, y_train, y_test = b_prepare_data.get_train_test_data()

    # f_test_true = np.bincount(y_test) / wandb.config.num_bins

    def interim_eval_cb(y_test_pred):
        d_evaluate.evaluate(y_test, y_test_pred)

    print("Training model…")
    y_test_pred = c_dsea.run(X_train, X_test, y_train, interim_eval_cb)

    print("Evaluating model…")
    d_evaluate.evaluate(y_test, y_test_pred, save=True)


if __name__ == '__main__':
    run()
