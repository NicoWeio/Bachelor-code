import b_prepare_data
import c_dsea
import d_evaluate
import x_config
from sklearn.model_selection import train_test_split
import wandb

wandb.init(project="dsea-corn", config=x_config.config)


def run():
    print("Loading data…")
    X, y = b_prepare_data.get_data(dummy=False,
                                   #    to_numpy=True,
                                   nrows=wandb.config.nrows,
                                   )
    # y = y.astype(np.int64)  # convert category → int64

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # f_test_true = np.bincount(y_test) / wandb.config.num_bins

    print("Training model…")
    y_test_pred = c_dsea.run(X_train, X_test, y_train)

    print("Evaluating model…")
    d_evaluate.evaluate(y_test, y_test_pred, save=True)


if __name__ == '__main__':
    run()
