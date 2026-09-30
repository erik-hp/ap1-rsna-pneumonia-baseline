import pandas as pd

from rsna_baseline.study import (
    build_ablation_families,
    build_error_cases,
    dataset_audit,
)


def _df():
    return pd.DataFrame(
        {
            "patientId": [f"p{i}" for i in range(6)],
            "Target": [0, 1, 0, 1, 0, 1],
            "hog_0": [0.1, 0.8, 0.2, 0.9, 0.3, 0.7],
            "lbp_0": [1, 2, 1, 2, 1, 2],
            "glcm_0": [5, 6, 5, 6, 5, 6],
        }
    )


def test_auto_ablation_builds_all_non_empty_combinations():
    families = build_ablation_families(_df())
    assert set(families) == {
        "HOG",
        "LBP",
        "GLCM",
        "HOG+LBP",
        "HOG+GLCM",
        "LBP+GLCM",
        "HOG+LBP+GLCM",
    }


def test_dataset_audit_reports_prevalence_and_feature_counts():
    df = _df()
    families = build_ablation_families(df)
    audit = dataset_audit(df, "patientId", "Target", families)
    assert audit["rows"] == 6
    assert audit["unique_groups"] == 6
    assert audit["positive_prevalence"] == 0.5
    assert audit["feature_count_total"] == 3


def test_error_cases_selects_false_positive_and_false_negative():
    predictions = pd.DataFrame(
        {
            "feature_family": ["HOG"] * 4,
            "model": ["svm_linear"] * 4,
            "y_true": [0, 1, 1, 0],
            "y_pred": [1, 0, 1, 0],
            "y_score": [0.9, -0.8, 0.7, -0.5],
        }
    )
    errors = build_error_cases(predictions, top_per_type=2)
    assert set(errors["error_type"]) == {"FP", "FN"}
