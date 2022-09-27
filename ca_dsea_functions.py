from cherenkovdeconvolution.stepsize import alpha_adaptive_run
from cherenkovdeconvolution.discretize import TreeDiscretizer
import wandb


def get_adaptive_alpha(X_train, X_test, y_train):
    discretizer = TreeDiscretizer(
        X_train, y_train,
        # Interpret the J leaves of a decision tree as clusters.
        J=wandb.config.J,
        # This value is proportional to the number of bins, based on a recommendation by Mirko Bunse.
        # J=wandb.config.J_factor * wandb.config.num_bins,
        seed=42,
    )
    X_data_discrete = discretizer.discretize(X_test)
    X_train_discrete = discretizer.discretize(X_train)
    Y_BINS = list(range(wandb.config.num_bins))  # TODO
    alpha = alpha_adaptive_run(
        X_data_discrete, X_train_discrete, y_train,
        tau=0,
        bins_y=Y_BINS,
    )
    return alpha
