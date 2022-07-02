from cherenkovdeconvolution import dsea
import numpy as np
from sklearn.model_selection import train_test_split
#
import b_prepare_data
import c_corn
from x_config import *


print("Loading data…")
X, y = b_prepare_data.get_data(dummy=False,
                               to_numpy=True,
                               nrows=NROWS,
                               )
# y = y.astype(np.int64)  # convert category → int64

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = c_corn.CornClassifier()

# classifier.fit(X, y)

f_est = dsea(X_test, X_train, y_train, classifier)
