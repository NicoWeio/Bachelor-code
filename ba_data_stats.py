import numpy as np


def stats_singleset(X, y):
    lines = {
        "events": f"{len(X)}",
        "events in the smallest bin": f"{np.min(np.bincount(y))}",
        "events in the largest bin": f"{np.max(np.bincount(y))}",
    }
    return lines


def stats(X_train, X_test, y_train, y_test):
    stats_train = stats_singleset(X_train, y_train)
    stats_test = stats_singleset(X_test, y_test)

    # combined_lines = {k: f"{stats_train[k]} / {stats_test[k]}" for k in stats_train}

    return (
        "Train: \n" +
        '\n'.join([f"- {k}: {v}" for k, v in stats_train.items()]) +
        "\nTest: \n" +
        '\n'.join([f"- {k}: {v}" for k, v in stats_test.items()])
    )
