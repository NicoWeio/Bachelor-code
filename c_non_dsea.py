import logging
import numpy as np
import wandb
#
import c_corn

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def run(X_train, X_test, y_train):
    classifier = c_corn.CornClassifier(
        input_size=X_train.shape[1],
        num_classes=np.bincount(y_train).shape[0],
    )

    classifier.fit(X_train, y_train)

    probas = classifier.predict_proba(X_test)

    return probas



# y_test_pred = classifier.predict_proba(X_test)
# f_test_pred = np.bincount(y_test) / wandb.config.num_bins

# probas = y_test_pred


# dist = util.chi2s(f_test_true, f_test_pred)
# print(f"Chi2 distance: {dist:.10f}")
