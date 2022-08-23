import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score
from scipy.stats import wasserstein_distance
from da_evaluate_plots import *
from da_evaluate_plots import plot_spectrum
from x_config import config  # TODO!
from cherenkovdeconvolution.util import chi2s

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

    metrics = {
        # 'accuracy': np.mean(true_labels == np.argmax(predicted_probas, axis=1)),
        'accuracy': accuracy_score(true_labels, np.argmax(predicted_probas, axis=1)),
        'chi2': chi2s(true_spectrum, pred_spectrum),
        # TODO: Which average mode to choose?
        'jaccard': jaccard_score(true_labels, np.argmax(predicted_probas, axis=1), average='micro'),
        'wd': wasserstein_distance(true_spectrum, pred_spectrum),
    }
    DATASET = 'test'
    metrics = {f'{k}_{DATASET}': v for k, v in metrics.items()}

    print(*[f"{k}: {v:.4f}" for k, v in metrics.items()], sep='\n')
    wandb.log(metrics)

    # ███ Plots
    plot_spectrum(true_spectrum, pred_spectrum, BINS, save=save)
    plot_single_events(true_labels, predicted_probas, BINS, save=save)


def save_bootstrap(bs_bundle):
    # Save the bootstrap bundle for later analysis, using np.savez
    for i, (y_test, y_test_pred) in enumerate(bs_bundle):
        np.savez(f'build/bootstrap/{i}', y_test=y_test, y_test_pred=y_test_pred)
    print(f"Saved {len(bs_bundle)} bootstrap bundles.")


def load_bootstrap():
    # Load the bootstrap bundle for later analysis, using np.load
    bs_bundle = []
    from pathlib import Path
    for bs_file in Path('build/bootstrap').glob('*.npz'):
        bs_data = np.load(bs_file)
        y_test = bs_data['y_test']
        y_test_pred = bs_data['y_test_pred']
        bs_bundle.append((y_test, y_test_pred))
    assert len(bs_bundle) > 0
    return bs_bundle


def evaluate_bootstrap(bs_bundle, save=False):
    if save:
        save_bootstrap(bs_bundle)

    per_bin_differences = []
    for true_labels, predicted_probas in bs_bundle:
        true_spectrum = spectrum_from_labels(true_labels)
        pred_spectrum = spectrum_from_probas(predicted_probas)
        per_bin_difference = true_spectrum - pred_spectrum  # TODO: abs()?
        per_bin_differences.append(per_bin_difference)

    per_bin_differences = np.array(per_bin_differences)
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
