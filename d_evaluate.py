import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from x_config import *

print("Loading data…")
eval_df = pd.read_hdf('build_large/eval.hdf5', key='eval')
# labels, predicted_labels = eval_df[['labels', 'predicted_labels']]
labels = eval_df['labels']
predicted_labels = eval_df['predicted_labels']
predicted_probas = np.array(eval_df['predicted_probas'].to_list())

# print(eval_df)
# print(labels[:10])
# print(labels, predicted_labels)

BINS = np.arange(NUM_BINS)


def spectrum_from_labels(labels):
    return np.bincount(labels, minlength=NUM_BINS)


def spectrum_from_probas(probas):
    assert probas.shape[1] == NUM_BINS
    return np.sum(probas, axis=0)


# █ Energy distribution
true_spectrum = spectrum_from_labels(labels)
pred_spectrum = spectrum_from_probas(predicted_probas)
pred_spectrum_class = spectrum_from_labels(predicted_labels)

fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# plt.hist(BINS[:-1], BINS, weights=spectrum_from_labels(labels), color='red', label='true class')

axs[0].plot(BINS, true_spectrum, drawstyle='steps-mid', color='red', zorder=10, linewidth=3, label='true class')
# axs[0].plot(BINS, spectrum_from_labels(predicted_labels), drawstyle='steps-mid', color='royalblue', label='predicted class')
axs[0].plot(BINS, pred_spectrum, drawstyle='steps-mid', color='green', label='predicted probas')
axs[0].set_ylabel('count')
axs[0].set_yscale('log')

axs[1].bar(BINS, (pred_spectrum - true_spectrum) / true_spectrum, label="relative deviation")

for ax in axs:
    ax.set_xlabel('class')
    ax.set_xticks(np.arange(NUM_BINS))
    ax.grid()
    ax.legend()

plt.tight_layout()
plt.savefig(f'build/corn__hist_log_{run_id()}.pdf')
plt.savefig(f'build/corn__hist_log_{run_id()}.png')
# wandb_run.log({"hist_log": plt})
# plt.show()


# █ Single events
SINGLE_EVENTS_GRIDSIZE = (4, 4)
SINGLE_EVENTS_NUM = SINGLE_EVENTS_GRIDSIZE[0] * SINGLE_EVENTS_GRIDSIZE[1]

# → one sample for each class
events_per_class = SINGLE_EVENTS_NUM // NUM_BINS # not rounding up so some subplots will be empty
# events_per_class = SINGLE_EVENTS_NUM // NUM_BINS + 1 # rounding up so the plot is filled
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
    ax.plot(BINS, predicted_probas[sample_i], drawstyle='steps-mid', color='green', label='predicted class')
    ax.set_xlabel('class')
    ax.set_ylabel('count')
    ax.set_yscale('log')
    ax.grid()
    ax.legend()
plt.tight_layout()
plt.savefig(f'build/corn__single_events_{run_id()}.pdf')
plt.savefig(f'build/corn__single_events_{run_id()}.png')


# █ Confusion matrix
# TODO: Use predicted_probas instead of predicted_labels
confusion_mtx = confusion_matrix(labels, predicted_labels)
df_cm = pd.DataFrame(confusion_mtx/confusion_mtx.sum(axis=1)[:, np.newaxis])
plt.figure(figsize=(12, 10))
ax = sns.heatmap(df_cm, annot=True, cmap='coolwarm')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix (normalized)')
plt.savefig(f'build/corn__confusion_{run_id()}.pdf')
plt.savefig(f'build/corn__confusion_{run_id()}.png')
# wandb_run.log({"confusion": plt})
