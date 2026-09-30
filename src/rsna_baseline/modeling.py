from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class ModelSpec:
    estimator: BaseEstimator
    param_grid: dict[str, list[Any]]


def build_model_specs(random_state: int = 42) -> dict[str, ModelSpec]:
    """Modelos clássicos e grades pequenas o suficiente para o TP."""
    return {
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
                            class_weight="balanced",
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
                            class_weight="balanced",
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
                class_weight="balanced_subsample",
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
                class_weight="balanced",
                random_state=random_state,
            ),
            param_grid={
                "learning_rate": [0.05, 0.1],
                "max_leaf_nodes": [15, 31],
                "l2_regularization": [0.0, 1.0],
            },
        ),
    }


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
