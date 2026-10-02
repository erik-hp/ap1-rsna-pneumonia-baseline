import math

import pandas as pd

from rsna_baseline.evaluation import (
    box_iou,
    classification_metrics,
    detection_ap,
    froc,
    mean_ap,
)


def test_box_iou_identical_and_disjoint():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


def test_classification_metrics_perfect():
    metrics = classification_metrics(
        y_true=[0, 0, 1, 1],
        y_score=[0.1, 0.2, 0.8, 0.9],
        y_pred=[0, 0, 1, 1],
    )
    for key, value in metrics.items():
        assert math.isclose(value, 1.0), (key, value)


def test_detection_ap_perfect():
    gt = pd.DataFrame(
        [
            {"patientId": "a", "x": 0, "y": 0, "width": 10, "height": 10},
            {"patientId": "b", "x": 20, "y": 20, "width": 5, "height": 5},
        ]
    )
    pred = gt.copy()
    pred["score"] = [0.9, 0.8]
    assert math.isclose(detection_ap(gt, pred, iou_threshold=0.5), 1.0)
    assert math.isclose(mean_ap(gt, pred)["map_50_95"], 1.0)


def test_froc_perfect():
    gt = pd.DataFrame(
        [{"patientId": "a", "x": 0, "y": 0, "width": 10, "height": 10}]
    )
    pred = gt.copy()
    pred["score"] = [0.9]
    values = froc(gt, pred, image_ids=["a", "negative"])
    assert values["froc_sens_at_0.125_fp_per_image"] == 1.0


def test_froc_tied_scores_respect_threshold_semantics():
    gt = pd.DataFrame(
        [{"patientId": "a", "x": 0, "y": 0, "width": 10, "height": 10}]
    )
    pred = pd.DataFrame(
        [
            {
                "patientId": "a",
                "x": 0,
                "y": 0,
                "width": 10,
                "height": 10,
                "score": 0.9,
            },
            {
                "patientId": "negative",
                "x": 20,
                "y": 20,
                "width": 5,
                "height": 5,
                "score": 0.9,
            },
        ]
    )
    values = froc(
        gt,
        pred,
        image_ids=["a", "negative"],
        fp_per_image_points=(0.25, 0.5),
    )
    assert values["froc_sens_at_0.25_fp_per_image"] == 0.0
    assert values["froc_sens_at_0.5_fp_per_image"] == 1.0


def test_froc_empty_predictions_returns_zero_sensitivity():
    gt = pd.DataFrame(
        [{"patientId": "a", "x": 0, "y": 0, "width": 10, "height": 10}]
    )
    pred = pd.DataFrame(
        columns=["patientId", "x", "y", "width", "height", "score"]
    )
    values = froc(gt, pred, image_ids=["a"])
    assert all(value == 0.0 for value in values.values())
