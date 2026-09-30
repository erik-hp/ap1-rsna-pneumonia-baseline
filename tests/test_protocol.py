import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def test_stratified_group_kfold_has_no_group_overlap():
    groups = pd.Series([f"p{i}" for i in range(10) for _ in range(2)])
    y = pd.Series([i % 2 for i in range(10) for _ in range(2)])
    X = np.zeros((len(y), 1))

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in cv.split(X, y, groups):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups)
