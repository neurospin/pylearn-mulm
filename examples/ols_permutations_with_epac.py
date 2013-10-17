"""
Permutation example using epac

@author: jinpeng.li@cea.fr

"""

import numpy as np
import random
from epac import LocalEngine, SomaWorkflowEngine
from epac import Perms
from epac import ColumnSplitter
from mulm import MUOLSStatsPredictions
from mulm import PValR2Reducer


if __name__ == "__main__":
    # =============================================================================
    # Generate data
    n_samples = 10
    n_xfeatures = 20
    n_yfeatures = 15
    x_n_groups = 3
    y_n_groups = 2
    X = np.random.randn(n_samples, n_xfeatures)
    Y = np.random.randn(n_samples, n_yfeatures)
    X_group_indices = np.array([random.randint(0, x_n_groups)\
        for i in xrange(n_xfeatures)])

    # =============================================================================
    # Build EPAC Workflow in two steps
    col_splitter = ColumnSplitter(MUOLSStatsPredictions(),
                              {"X": X_group_indices})
    row_perms = Perms(col_splitter,
                  n_perms=10,
                  reducer=PValR2Reducer(),
                  permute="X",
                  col_or_row=False,
                  need_group_key=False)
    row_perms.run(X=X, Y=Y)            # map
    results = row_perms.reduce()       # reduce
    print results