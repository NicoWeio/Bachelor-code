import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from x_config import *

eval_df = pd.read_hdf('build_large/eval.hdf5', key='eval')
# labels, predicted_labels = eval_df[['labels', 'predicted_labels']]
labels = eval_df['labels']
predicted_labels = eval_df['predicted_labels']
predicted_probas = np.array(eval_df['predicted_probas'].to_list())

# print(eval_df)
# print(labels[:10])
# print(labels, predicted_labels)

def spectrum_from_labels(labels):
    return np.bincount(labels, minlength=NUM_BINS)

def spectrum_from_probas(probas):
    assert probas.shape[1] == NUM_BINS
    return np.sum(probas, axis=0)

# Energy distribution
plt.figure(figsize=(10, 6))
plt.bar(np.arange(NUM_BINS), spectrum_from_labels(labels), alpha=0.7, color='red', label='true class')
plt.bar(np.arange(NUM_BINS), spectrum_from_labels(predicted_labels), alpha=0.7, color='royalblue', label='predicted class')
plt.bar(np.arange(NUM_BINS), spectrum_from_probas(predicted_probas), alpha=0.7, color='green', label='sum_probas')
plt.xlabel('Class')
plt.ylabel('pdf')
plt.xticks(np.arange(NUM_BINS))
plt.grid()
plt.legend()
plt.savefig(f'build/corn__hist_log_{run_id()}.pdf')
plt.savefig(f'build/corn__hist_log_{run_id()}.png')
# wandb_run.log({"hist_log": plt})
# plt.show()

# Confusion matrix
confusion_mtx = confusion_matrix(labels, predicted_labels)
df_cm = pd.DataFrame(confusion_mtx/confusion_mtx.sum(axis=1)[:, np.newaxis])
plt.figure(figsize=(12,10))
ax = sns.heatmap(df_cm, annot=True, cmap='coolwarm')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix (normalized)')
plt.savefig(f'build/corn__confusion_{run_id()}.pdf')
plt.savefig(f'build/corn__confusion_{run_id()}.png')
# wandb_run.log({"confusion": plt})
