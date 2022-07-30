from cherenkovdeconvolution import dsea
import logging
import numpy as np
import wandb
#
import c_corn
from x_config import config as default_config

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def run(X_train, X_test, y_train):
    classifier = c_corn.CornClassifier(
        input_size=X_train.shape[1],
        num_classes=np.bincount(y_train).shape[0],
    )

    def dsea_callback(f, k, alpha, chi2s):
        """
        f: prior
        k: iteration
        alpha: step size
        chi2s: Chi Square distance between iterations
        """
        wandb.log({'f': f, 'k': k, 'alpha': alpha, 'chi2s_iters': chi2s})

        print("▒"*10)
        print(f"Iteration {k} of {wandb.config.num_dsea_iterations}: alpha = {alpha:.3f}, chi2s_iters = {chi2s:.4f}")
        print(f"f = {f}")
        print()

    f_est, probas = dsea(X_test,
                         X_train,
                         y_train,
                         classifier,
                         inspect=dsea_callback,
                         return_contributions=True,
                         K=wandb.config.num_dsea_iterations,
                         )

    return probas

# if __name__ == "__main__":
#     print("Saving eval HDF5…")
#     # Export for evaluation
#     eval_df = pd.DataFrame({
#         'labels': y_test,
#         # 'predicted_labels': all_predicted_labels,
#         'predicted_probas': probas.tolist()
#     })
#     # print("Saving eval CSV…")
#     # eval_df.to_csv('build_large/eval.csv', index=False)
#     eval_df.to_hdf('build_large/eval.hdf5', key='eval', index=False)
