import numpy as np
import pandas as pd

from rsna_baseline.detection import (
    group_stratified_row_splits,
    non_max_suppression,
    sample_training_regions,
)


def test_group_stratified_row_splits_keep_patients_disjoint():
    groups = pd.Series(np.repeat([f"p{i}" for i in range(8)], 3))
    labels = {f"p{i}": int(i >= 4) for i in range(8)}

    splits = group_stratified_row_splits(groups, labels, n_splits=2, seed=42)

    for train_idx, test_idx in splits:
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert not train_groups.intersection(test_groups)
        assert {labels[g] for g in train_groups} == {0, 1}
        assert {labels[g] for g in test_groups} == {0, 1}


def test_sample_training_regions_limits_negatives_and_keeps_groups():
    rows = []
    for patient in ["p0", "p1", "p2", "p3"]:
        for i in range(10):
            rows.append(
                {
                    "patientId": patient,
                    "Target": int(patient in {"p2", "p3"} and i < 2),
                    "f": float(i),
                }
            )
    frame = pd.DataFrame(rows)

    sampled = sample_training_regions(
        frame,
        max_neg_pos_ratio=2.0,
        seed=42,
    )

    positives = int((sampled.Target == 1).sum())
    negatives = int((sampled.Target == 0).sum())
    assert positives == 4
    assert negatives <= max(positives * 2, frame.patientId.nunique())
    assert set(sampled.patientId) == set(frame.patientId)


def test_nms_suppresses_overlapping_boxes():
    predictions = pd.DataFrame(
        [
            {"patientId": "p1", "x": 0, "y": 0, "width": 100, "height": 100, "score": 0.9},
            {"patientId": "p1", "x": 5, "y": 5, "width": 100, "height": 100, "score": 0.8},
            {"patientId": "p1", "x": 150, "y": 150, "width": 50, "height": 50, "score": 0.7},
        ]
    )

    kept = non_max_suppression(
        predictions,
        iou_threshold=0.3,
        top_k=20,
    )

    assert len(kept) == 2
    assert kept.score.tolist() == [0.9, 0.7]
