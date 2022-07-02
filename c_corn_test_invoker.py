import numpy as np
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

classifier = c_corn.CornClassifier()

classifier.fit(X, y)
