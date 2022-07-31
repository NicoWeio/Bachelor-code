import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import wasserstein_distance
from da_evaluate_plots import *
from da_evaluate_plots import plot_spectrum
from x_config import config  # TODO!

NUM_BINS = config['num_bins']
BINS = np.arange(NUM_BINS)


def spectrum_from_labels(labels, norm=True):
    spectrum = np.bincount(labels, minlength=NUM_BINS).astype(float)
    if norm:
        spectrum /= len(labels)
    return spectrum


def spectrum_from_probas(probas, norm=True):
    assert probas.shape[1] == NUM_BINS
    spectrum = np.sum(probas, axis=0)
    if norm:
        spectrum /= len(probas)
    return spectrum


def evaluate(true_labels, predicted_probas, save=False):
    """
    Evaluation & Plots.
    To be called on the test set.
    """
    # █ Verification
    # → Every bin is represented in the (true) `labels`
    assert np.all(np.isin(BINS, true_labels)), "Not all bins are represented in the labels!"
    print("Verification passed.")

    # █ Energy distribution
    true_spectrum = spectrum_from_labels(true_labels)
    pred_spectrum = spectrum_from_probas(predicted_probas)
    # pred_spectrum_class = spectrum_from_labels(predicted_labels)

    print("Wasserstein distance:", wasserstein_distance(true_spectrum, pred_spectrum))
    wandb.log({'wd_test': wasserstein_distance(true_spectrum, pred_spectrum)})

    # ███ Plots
    plot_spectrum(true_spectrum, pred_spectrum, BINS, save=save)


if __name__ == '__main__':
    print("Loading data…")
    eval_df = pd.read_hdf('build_large/eval.hdf5', key='eval')
    # labels, predicted_labels = eval_df[['labels', 'predicted_labels']]
    labels = eval_df['labels']
    # predicted_labels = eval_df['predicted_labels']
    predicted_probas = np.array(eval_df['predicted_probas'].to_list())

    evaluate(labels, predicted_probas)
