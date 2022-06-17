import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
plt.xlabel('Class')
plt.ylabel('pdf')
plt.xticks(np.arange(NUM_BINS))
plt.grid()
plt.legend()
plt.savefig(f'build/corn__hist_log_{run_id()}.pdf')
plt.savefig(f'build/corn__hist_log_{run_id()}.png')
# wandb_run.log({"hist_log": plt})
# plt.show()
