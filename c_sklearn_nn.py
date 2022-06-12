# NOTE: This served as a first test. It does not povide reproducible results (for an unknown reason).

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# read data
df = pd.read_csv('build_large/data.csv', nrows=50000)  # TODO nrows

# discretize the target neutrino energy

# Variables: Label
# drop out all events with Energies outside the range
lower_limit = 100
upper_limit = 10**5

# Variables: NN
num_bins = 10  # number of bins (energy classes), output_shape

# throw out extreme high and low energy neutrinos
df = df[(df['MCPrimary.energy'] < upper_limit) & (df['MCPrimary.energy'] > lower_limit)]
print(df)

# log-scaled Binning
bins = np.logspace(np.log10(lower_limit), np.log10(upper_limit), num_bins+1)

# new column with discretized energies
df['E_discr'] = pd.cut(df['MCPrimary.energy'], bins=bins, labels=range(len(bins)-1))
print(df['E_discr'].value_counts())

# one hot encoded vector (necessary for cce)
# Convert categorical variable into dummy/indicator variables
df_E_dummy = pd.get_dummies(df['E_discr'])

X = df.drop(columns=['MCPrimary.energy', 'E_discr']).to_numpy()
y = df_E_dummy.to_numpy()

X_train, X_eval, y_train, y_eval = train_test_split(
    X, y,
    test_size=0.3, random_state=42)

print(f"X_train shape: {X_train.shape}", f"y_train shape: {y_train.shape}",
      f"X_eval shape: {X_eval.shape}", f"y_eval shape: {y_eval.shape}")

########################################################################################################################

# use a NN from sklearn for now
classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=5000, random_state=42)

classifier.fit(X_train, y_train)

print(f"Training set score: {classifier.score(X_train, y_train)}")
print(f"Evaluation set score: {classifier.score(X_eval, y_eval)}")

########################################################################################################################

# propability for each class
y_eval_pred = classifier.predict(X_eval)

# choose class with max value
y_pred_max = np.zeros_like(y_eval_pred)
y_pred_max[np.arange(len(y_eval_pred)), y_eval_pred.argmax(1)] = 1

# print a histogram with the true/predicted energy classes
plt.figure(figsize=(10, 6))
plt.bar(np.arange(10), y_eval.sum(axis=0)/y_eval.sum(), alpha=0.7, color='red', label='true class')
plt.bar(np.arange(10), y_pred_max.sum(axis=0)/y_eval.sum(), alpha=0.7, color='royalblue', label='predicted class')

plt.legend()
plt.grid()
plt.xlabel('Class')
plt.ylabel('pdf')
plt.xticks(np.arange(10))
plt.savefig('build/sklearn_nn__hist_log.pdf')
# plt.show()
