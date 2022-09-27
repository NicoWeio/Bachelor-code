import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.metrics import (accuracy_score, confusion_matrix, jaccard_score,
                             mean_absolute_error, mean_squared_error)

from cherenkovdeconvolution.util import chi2s
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


def get_metrics(true_labels, predicted_probas):
    true_spectrum = spectrum_from_labels(true_labels)
    pred_spectrum = spectrum_from_probas(predicted_probas)

    return {
        # 'accuracy': np.mean(true_labels == np.argmax(predicted_probas, axis=1)),
        'accuracy': accuracy_score(true_labels, np.argmax(predicted_probas, axis=1)),
        'chi2': chi2s(true_spectrum, pred_spectrum),
        # TODO: Which average mode to choose?
        'jaccard': jaccard_score(true_labels, np.argmax(predicted_probas, axis=1), average='micro'),
        'wd': wasserstein_distance(true_spectrum, pred_spectrum),
        # TODO 'mae': mean_absolute_error(true_spectrum, pred_spectrum),
        'rmse': mean_squared_error(true_spectrum, pred_spectrum, squared=False),
    }


def evaluate(true_labels, predicted_probas, save=False, log=True, dataset='test') -> dict:
    """
    Evaluation & Plots.
    To be called on the test set.
    """
    # █ Verification
    # → Every bin is represented in the (true) `labels`
    assert np.all(np.isin(BINS, true_labels)), "Not all bins are represented in the labels!"
    if log:
        print("Verification passed.")

    # █ Energy distribution
    true_spectrum = spectrum_from_labels(true_labels)
    pred_spectrum = spectrum_from_probas(predicted_probas)
    # pred_spectrum_class = spectrum_from_labels(predicted_labels)

    metrics = get_metrics(true_labels, predicted_probas)
    metrics = {f'{dataset}/{k}': v for k, v in metrics.items()}

    if log:
        print(*[f"{k}: {v:.4f}" for k, v in metrics.items()], sep='\n')
        wandb.log(metrics)

    # █ Plots
    if save:
        plot_spectrum(true_spectrum, pred_spectrum, BINS, save=save)
        plot_single_events(true_labels, predicted_probas, BINS, save=save)

    return metrics


def save_bootstrap(y_test, y_test_pred, index):
    # Save the bootstrap bundle for later analysis, using np.savez
    np.savez(f'build_large/bootstrap/{index}', y_test=y_test, y_test_pred=y_test_pred)


def load_bootstrap_bundle(subdir=''):
    """
    Load a bundle of bootstraps for analysis, using np.load

    Returns
    -------
    bs_bundle : list of (y_test, y_test_pred) tuples
    """
    bs_bundle = []
    from pathlib import Path
    for bs_file in Path(f"build_large/bootstrap/{subdir}").glob('*.npz'):
        bs_data = np.load(bs_file)
        y_test = bs_data['y_test']
        y_test_pred = bs_data['y_test_pred']
        bs_bundle.append((y_test, y_test_pred))
    assert len(bs_bundle) > 0
    return bs_bundle


def evaluate_bootstrap(bs_bundle):
    true_spectra = []
    pred_spectra = []
    for true_labels, predicted_probas in bs_bundle:
        true_spectra.append(spectrum_from_labels(true_labels))
        pred_spectra.append(spectrum_from_probas(predicted_probas))
    true_spectra = np.array(true_spectra)
    pred_spectra = np.array(pred_spectra)

    wandb.summary['bootstrap/true_spectra'] = true_spectra.tolist()
    wandb.summary['bootstrap/pred_spectra'] = pred_spectra.tolist()

    per_bin_differences = true_spectra - pred_spectra  # TODO: abs()?
    mean_per_bin_difference = np.mean(per_bin_differences, axis=0)
    std_per_bin_difference = np.std(per_bin_differences, axis=0)

    with np.printoptions(precision=3):
        print(f"FOOOOO {mean_per_bin_difference} ± {std_per_bin_difference}")

    # █ Energy distribution

    # ███ Plots
    # import uncertainties.unumpy as unp


if __name__ == '__main__':
    print("Loading data…")
    eval_df = pd.read_hdf('build_large/eval.hdf5', key='eval')
    # labels, predicted_labels = eval_df[['labels', 'predicted_labels']]
    labels = eval_df['labels']
    # predicted_labels = eval_df['predicted_labels']
    predicted_probas = np.array(eval_df['predicted_probas'].to_list())

    evaluate(labels, predicted_probas)
