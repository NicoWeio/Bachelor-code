import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from x_config import config  # TODO!
from x_config import run_id as _run_id

run_id = lambda: _run_id(config)


def plot_spectrum(true_spectrum, pred_spectrum, BINS, NUM_BINS):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # plt.hist(BINS[:-1], BINS, weights=spectrum_from_labels(labels), color='red', label='true class')

    axs[0].plot(BINS, true_spectrum, drawstyle='steps-mid', color='red', linewidth=2, label='true class')
    # axs[0].plot(BINS, spectrum_from_labels(predicted_labels), drawstyle='steps-mid', color='royalblue', label='predicted class')
    axs[0].plot(BINS, pred_spectrum, drawstyle='steps-mid', color='green', zorder=10, label='predicted probas')
    axs[0].set_ylabel('count')
    axs[0].set_yscale('log')

    axs[1].bar(BINS, (pred_spectrum - true_spectrum) / true_spectrum, label="relative deviation")

    for ax in axs:
        ax.set_xlabel('class')
        ax.set_xticks(np.arange(NUM_BINS))
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'build/spectrum_{run_id()}.pdf')
    plt.savefig(f'build/spectrum_{run_id()}.png')
    # wandb_run.log({"hist_log": plt})
    # plt.show()


# █ Single events
def plot_single_events():
    SINGLE_EVENTS_GRIDSIZE = (4, 4)
    SINGLE_EVENTS_NUM = SINGLE_EVENTS_GRIDSIZE[0] * SINGLE_EVENTS_GRIDSIZE[1]

    # → one sample for each class
    events_per_class = SINGLE_EVENTS_NUM // NUM_BINS  # not rounding up so some subplots will be empty
    # events_per_class = SINGLE_EVENTS_NUM // NUM_BINS + 1  # rounding up so all subplots are filled
    sample_indices = []
    for i in range(NUM_BINS):
        possible_indices = np.where(labels == i)[0]
        sample_indices += np.random.choice(possible_indices, events_per_class, replace=False).tolist()

    # → random samples
    # sample_indices = np.random.choice(len(labels), SINGLE_EVENTS_NUM, replace=False).tolist()

    fig, axs = plt.subplots(
        # SINGLE_EVENTS_GRIDSIZE[0], SINGLE_EVENTS_GRIDSIZE[1],
        *SINGLE_EVENTS_GRIDSIZE,
        figsize=(SINGLE_EVENTS_GRIDSIZE[1] * 5, SINGLE_EVENTS_GRIDSIZE[0] * 5)
    )
    for i, ax in enumerate(axs.flat):
        if i >= len(sample_indices):
            ax.axis('off')
            continue

        sample_i = sample_indices[i]
        ax.axvline(labels[sample_i], color='red', linestyle='--', label='true class')
        ax.plot(BINS, predicted_probas[sample_i], drawstyle='steps-mid', color='green', label='predicted probas')
        ax.set_xlabel('class')
        ax.set_ylabel('probability')
        ax.set_yscale('log')
        ax.grid()
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'build/single_events_{run_id()}.pdf')
    plt.savefig(f'build/single_events_{run_id()}.png')


# █ Per-bin spectra
def plot_per_bin_spectra():
    GRIDSIZE = (3, 4)
    fig, axs = plt.subplots(
        *GRIDSIZE,
        figsize=(GRIDSIZE[1] * 5, GRIDSIZE[0] * 5)
    )
    # TODO: iterate over bins instead of axes for clarity
    for i, ax in enumerate(axs.flat):
        if i >= NUM_BINS:
            ax.axis('off')
            continue

        ax.axvline(i, color='red', linestyle='--', label='true class')
        ax.plot(BINS, spectrum_from_probas(predicted_probas[labels == i]),
                drawstyle='steps-mid', color='green', label='predicted probas (→ spectrum)')
        ax.set_xlabel('class')
        ax.set_ylabel('count')
        ax.set_yscale('log')
        ax.grid()
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'build/per_bin_spectra_{run_id()}.pdf')
    plt.savefig(f'build/per_bin_spectra_{run_id()}.png')


# def plot_confusion_matrix():
#     # TODO: Use predicted_probas instead of predicted_labels
#     confusion_mtx = confusion_matrix(labels, predicted_labels)
#     df_cm = pd.DataFrame(confusion_mtx/confusion_mtx.sum(axis=1)[:, np.newaxis])
#     plt.figure(figsize=(12, 10))
#     ax = sns.heatmap(df_cm, annot=True, cmap='coolwarm')
#     bottom, top = ax.get_ylim()
#     ax.set_ylim(bottom + 0.5, top - 0.5)
#     plt.xlabel('Predicted label')
#     plt.ylabel('True label')
#     plt.title('Confusion Matrix (normalized)')
#     plt.savefig(f'build/confusion_{run_id()}.pdf')
#     plt.savefig(f'build/confusion_{run_id()}.png')
#     wandb_run.log({"confusion": plt})
