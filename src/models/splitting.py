"""Train/validation/test theo công ty, giữ JD cùng nguồn trong cùng tập."""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def assign_company_splits(frame, random_state=42):
    if frame.company_group.nunique() < 5:
        raise ValueError("Cần ít nhất 5 nhóm công ty để chia train/validation/test")
    labels = np.full(len(frame), "train", dtype=object)
    train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state).split(frame, groups=frame.company_group))
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state + 1).split(
        frame.iloc[train_val], groups=frame.iloc[train_val].company_group))
    labels[test] = "test"
    labels[train_val[validation]] = "validation"
    return labels
