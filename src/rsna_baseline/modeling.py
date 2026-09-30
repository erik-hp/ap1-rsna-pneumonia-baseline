from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class ModelSpec:
    estimator: BaseEstimator
    param_grid: dict[str, list[Any]]


def _with_optional_pca(
    specs: dict[str, ModelSpec],
    pca_variance: float | None,
) -> dict[str, ModelSpec]:
    if pca_variance is None:
        return specs
    if not 0.0 < pca_variance < 1.0:
        raise ValueError("pca_variance deve estar entre 0 e 1, por exemplo 0.95.")

    wrapped: dict[str, ModelSpec] = {}
    for name, spec in specs.items():
        if name == "dummy":
            wrapped[name] = spec
            continue

        if isinstance(spec.estimator, Pipeline):
            steps = []
            for step_name, step in spec.estimator.steps:
                if step_name == "model":
                    steps.append(("pca", PCA(n_components=pca_variance)))
                steps.append((step_name, step))
            wrapped[name] = ModelSpec(Pipeline(steps), spec.param_grid)
            continue

        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=pca_variance)),
                ("model", spec.estimator),
            ]
        )
        grid = {f"model__{key}": value for key, value in spec.param_grid.items()}
        wrapped[name] = ModelSpec(pipeline, grid)

    return wrapped


def build_model_specs(
    random_state: int = 42,
    class_weight_mode: str = "balanced",
    pca_variance: float | None = None,
) -> dict[str, ModelSpec]:
    """Modelos clássicos, com balanceamento e PCA opcionais para ablação."""
    if class_weight_mode not in {"balanced", "none"}:
        raise ValueError("class_weight_mode deve ser 'balanced' ou 'none'.")

    balanced = class_weight_mode == "balanced"
    svm_weight = "balanced" if balanced else None
    rf_weight = "balanced_subsample" if balanced else None
    hgb_weight = "balanced" if balanced else None

    specs = {
        "dummy": ModelSpec(
            estimator=DummyClassifier(strategy="prior"),
            param_grid={},
        ),
        "svm_linear": ModelSpec(
            estimator=Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        SVC(
                            kernel="linear",
                            class_weight=svm_weight,
                            probability=False,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            param_grid={"model__C": [0.1, 1.0, 10.0]},
        ),
        "svm_rbf": ModelSpec(
            estimator=Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            class_weight=svm_weight,
                            probability=False,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            param_grid={
                "model__C": [1.0, 10.0],
                "model__gamma": ["scale", 0.01],
            },
        ),
        "random_forest": ModelSpec(
            estimator=RandomForestClassifier(
                class_weight=rf_weight,
                random_state=random_state,
                n_jobs=-1,
            ),
            param_grid={
                "n_estimators": [200, 400],
                "max_depth": [None, 20],
                "min_samples_leaf": [1, 4],
            },
        ),
        "hist_gradient_boosting": ModelSpec(
            estimator=HistGradientBoostingClassifier(
                class_weight=hgb_weight,
                random_state=random_state,
            ),
            param_grid={
                "learning_rate": [0.05, 0.1],
                "max_leaf_nodes": [15, 31],
                "l2_regularization": [0.0, 1.0],
            },
        ),
    }
    return _with_optional_pca(specs, pca_variance)


def continuous_score(estimator: BaseEstimator, X):
    """Retorna um escore contínuo para ROC/PR sem exigir calibração."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        classes = list(getattr(estimator, "classes_", []))
        if proba.ndim == 2 and 1 in classes:
            return proba[:, classes.index(1)]
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()

    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)

    return estimator.predict(X)
