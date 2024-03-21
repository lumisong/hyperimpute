import numpy as np
from hyperimpute.plugins.imputers import Imputers

plugin = Imputers().get("hyperimpute")
plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])

