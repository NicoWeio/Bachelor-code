from cherenkovdeconvolution import dsea
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
import b_prepare_data
import c_corn
from x_config import *


print("Loading data…")
X, y = b_prepare_data.get_data(dummy=False,
                            #    to_numpy=True,
                               nrows=NROWS,
                               )
# y = y.astype(np.int64)  # convert category → int64

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = c_corn.CornClassifier(
    input_size=X.shape[1],
    num_classes=np.bincount(y).shape[0],
)

# classifier.fit(X, y)

f_est, probas = dsea(X_test, X_train, y_train, classifier, return_contributions=True)


# Export for evaluation
eval_df = pd.DataFrame({
    'labels': y_test,
    # 'predicted_labels': all_predicted_labels,
    'predicted_probas': probas.tolist()
})
# print("Saving eval CSV…")
# eval_df.to_csv('build_large/eval.csv', index=False)
print("Saving eval HDF5…")
eval_df.to_hdf('build_large/eval.hdf5', key='eval', index=False)
