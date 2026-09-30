from __future__ import annotations

from itertools import combinations
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn

from .io import ensure_columns, parse_families


def build_ablation_families(
    df: pd.DataFrame,
    base_spec: str = "HOG=hog_,LBP=lbp_,GLCM=glcm_",
) -> dict[str, list[str]]:
    """Gera todas as combinações não vazias das famílias base."""
    base = parse_families(base_spec, df)
    if len(base) < 2:
        raise ValueError("A ablação automática requer pelo menos duas famílias.")

    result: dict[str, list[str]] = {}
    names = list(base)
    for size in range(1, len(names) + 1):
        for combo in combinations(names, size):
            columns: list[str] = []
            for name in combo:
                for column in base[name]:
                    if column not in columns:
                        columns.append(column)
            result["+".join(combo)] = columns
    return result


def dataset_audit(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    families: dict[str, list[str]],
) -> dict:
    ensure_columns(df, [group_col, label_col])
    feature_columns = sorted({c for columns in families.values() for c in columns})
    ensure_columns(df, feature_columns)

    if df[label_col].isna().any():
        raise ValueError("Target contém valores ausentes.")
    if df[group_col].isna().any():
        raise ValueError("Identificador de paciente/grupo contém valores ausentes.")

    labels = set(pd.unique(df[label_col]))
    if labels - {0, 1}:
        raise ValueError("Target deve estar codificado como 0/1.")

    feature_frame = df[feature_columns]
    if feature_frame.isna().any().any():
        bad = feature_frame.columns[feature_frame.isna().any()].tolist()
        raise ValueError(f"Features contêm NaN: {bad[:10]}")

    values = feature_frame.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Features contêm valores infinitos.")

    class_counts = df[label_col].value_counts().sort_index()
    class_groups = df.groupby(label_col)[group_col].nunique().sort_index()

    box_columns = {"x", "y", "width", "height"}
    has_boxes = box_columns.issubset(df.columns)

    audit = {
        "rows": int(len(df)),
        "unique_groups": int(df[group_col].nunique()),
        "classes": {str(k): int(v) for k, v in class_counts.items()},
        "groups_per_class": {str(k): int(v) for k, v in class_groups.items()},
        "positive_prevalence": float((df[label_col] == 1).mean()),
        "feature_count_total": int(len(feature_columns)),
        "feature_counts_by_family": {k: int(len(v)) for k, v in families.items()},
        "duplicate_full_rows": int(df.duplicated().sum()),
        "rows_beyond_one_per_group": int(len(df) - df[group_col].nunique()),
        "has_candidate_boxes": bool(has_boxes),
    }

    if has_boxes:
        invalid_boxes = (
            (df["width"].fillna(0) <= 0)
            | (df["height"].fillna(0) <= 0)
            | df[["x", "y", "width", "height"]].isna().any(axis=1)
        )
        audit["invalid_candidate_boxes"] = int(invalid_boxes.sum())

    return audit


def build_error_cases(predictions: pd.DataFrame, top_per_type: int = 10) -> pd.DataFrame:
    required = {"feature_family", "model", "y_true", "y_pred", "y_score"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predições sem colunas necessárias: {sorted(missing)}")

    frame = predictions.copy()
    frame["error_type"] = np.select(
        [
            (frame["y_true"] == 0) & (frame["y_pred"] == 1),
            (frame["y_true"] == 1) & (frame["y_pred"] == 0),
            (frame["y_true"] == 1) & (frame["y_pred"] == 1),
        ],
        ["FP", "FN", "TP"],
        default="TN",
    )

    selected = []
    for (_, _), group in frame.groupby(["feature_family", "model"]):
        fp = group[group["error_type"] == "FP"].sort_values("y_score", ascending=False).head(top_per_type)
        fn = group[group["error_type"] == "FN"].sort_values("y_score", ascending=True).head(top_per_type)
        selected.extend([fp, fn])

    selected = [part for part in selected if not part.empty]
    if not selected:
        return frame.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    def cell(value) -> str:
        return str(value).replace("|", "\\|")

    headers = [cell(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def experiment_manifest(
    *,
    seed: int,
    outer_splits: int,
    inner_splits: int,
    scoring: str,
    class_weight_mode: str,
    pca_variance: float | None,
    auto_ablation: bool,
    feature_file: str,
    models: list[str],
    families: dict[str, list[str]],
) -> dict:
    return {
        "seed": seed,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "scoring": scoring,
        "class_weight_mode": class_weight_mode,
        "pca_variance": pca_variance,
        "auto_ablation": auto_ablation,
        "feature_file": Path(feature_file).name,
        "models": models,
        "families": {name: len(cols) for name, cols in families.items()},
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
