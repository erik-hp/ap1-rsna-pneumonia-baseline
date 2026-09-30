from sklearn.pipeline import Pipeline

from rsna_baseline.modeling import build_model_specs


def test_class_weight_can_be_disabled_for_sensitivity_analysis():
    specs = build_model_specs(class_weight_mode="none")
    svm = specs["svm_linear"].estimator.named_steps["model"]
    rf = specs["random_forest"].estimator
    assert svm.class_weight is None
    assert rf.class_weight is None


def test_optional_pca_stays_inside_model_pipeline():
    specs = build_model_specs(pca_variance=0.95)
    for name in ["svm_linear", "svm_rbf", "random_forest", "hist_gradient_boosting"]:
        estimator = specs[name].estimator
        assert isinstance(estimator, Pipeline)
        assert "pca" in estimator.named_steps
        assert "model" in estimator.named_steps
