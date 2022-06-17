import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from x_config import *

eval_df = pd.read_csv('build_large/eval.csv')
# labels, predicted_labels = eval_df[['labels', 'predicted_labels']]
labels = eval_df['labels']
predicted_labels = eval_df['predicted_labels']

# print(eval_df)
print(labels[:10])
# print(labels, predicted_labels)


# Energy distribution
plt.figure(figsize=(10, 6))
plt.bar(np.arange(NUM_BINS), np.bincount(labels), alpha=0.7, color='red', label='true class')
plt.bar(np.arange(NUM_BINS), np.bincount(predicted_labels, minlength=NUM_BINS), alpha=0.7, color='royalblue', label='predicted class')
# plt.bar(np.arange(NUM_BINS), np.bincount(predicted_labels, minlength=NUM_BINS), alpha=0.7, color='green', label='sum_probas')
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
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (normalized)')
plt.savefig(f'build/corn__confusion_{run_id()}.pdf')
plt.savefig(f'build/corn__confusion_{run_id()}.png')
# wandb_run.log({"confusion": plt})
