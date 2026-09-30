from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from .evaluation import classification_metrics
from .modeling import ModelSpec, continuous_score


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    outer_splits: int = 5
    inner_splits: int = 3
    scoring: str = "average_precision"
    n_jobs: int = -1


def _validate_protocol(y: pd.Series, groups: pd.Series, config: ExperimentConfig) -> None:
    if y.isna().any() or groups.isna().any():
        raise ValueError("Target e grupo/paciente não podem conter valores ausentes.")
    if set(pd.unique(y)) - {0, 1}:
        raise ValueError("Esta implementação espera alvo binário codificado como 0/1.")
    if y.nunique() < 2:
        raise ValueError("São necessárias as duas classes para avaliação.")
    if groups.nunique() < config.outer_splits:
        raise ValueError(
            f"Há apenas {groups.nunique()} grupos, menos que outer_splits={config.outer_splits}."
        )


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_specs: dict[str, ModelSpec],
    feature_family: str,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Validação cruzada aninhada.

    O CV externo estima desempenho. O CV interno escolhe hiperparâmetros
    exclusivamente dentro do treino externo.
    """
    _validate_protocol(y, groups, config)

    outer_cv = StratifiedGroupKFold(
        n_splits=config.outer_splits,
        shuffle=True,
        random_state=config.seed,
    )

    metrics_rows: list[dict] = []
    prediction_rows: list[dict] = []
    best_params_rows: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]

        overlap = set(groups_train).intersection(set(groups_test))
        if overlap:
            raise RuntimeError(f"Vazamento entre treino e teste no fold {fold}: {list(overlap)[:5]}")

        inner_cv = StratifiedGroupKFold(
            n_splits=config.inner_splits,
            shuffle=True,
            random_state=config.seed + fold,
        )

        for model_name, spec in model_specs.items():
            search = GridSearchCV(
                estimator=spec.estimator,
                param_grid=spec.param_grid,
                scoring=config.scoring,
                cv=inner_cv,
                refit=True,
                n_jobs=config.n_jobs,
                error_score="raise",
            )
            search.fit(X_train, y_train, groups=groups_train)

            best = search.best_estimator_
            y_score = np.asarray(continuous_score(best, X_test), dtype=float)
            y_pred = np.asarray(best.predict(X_test), dtype=int)
            fold_metrics = classification_metrics(y_test, y_score, y_pred)

            metrics_rows.append(
                {
                    "feature_family": feature_family,
                    "model": model_name,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_groups_train": groups_train.nunique(),
                    "n_groups_test": groups_test.nunique(),
                    **fold_metrics,
                }
            )

            for pos, row_index in enumerate(X_test.index):
                prediction_rows.append(
                    {
                        "row_index": row_index,
                        "feature_family": feature_family,
                        "model": model_name,
                        "fold": fold,
                        "y_true": int(y_test.iloc[pos]),
                        "y_pred": int(y_pred[pos]),
                        "y_score": float(y_score[pos]),
                    }
                )

            best_params_rows.append(
                {
                    "feature_family": feature_family,
                    "model": model_name,
                    "fold": fold,
                    "best_score_inner": float(search.best_score_),
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                }
            )

    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(prediction_rows),
        best_params_rows,
    )


def summarize_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        "auc_roc",
        "auc_pr",
        "sensitivity",
        "specificity",
        "f1",
        "balanced_accuracy",
    ]

    grouped = metrics.groupby(["feature_family", "model"])[metric_cols]
    numeric = grouped.agg(["mean", "std"]).reset_index()
    numeric.columns = [
        "_".join([str(p) for p in col if p]).rstrip("_") if isinstance(col, tuple) else col
        for col in numeric.columns
    ]

    rows = []
    for (family, model), frame in metrics.groupby(["feature_family", "model"]):
        row = {"feature_family": family, "model": model}
        for metric in metric_cols:
            mean = frame[metric].mean()
            std = frame[metric].std(ddof=1)
            row[metric] = f"{mean:.3f} ± {std:.3f}"
        rows.append(row)

    return numeric, pd.DataFrame(rows)
