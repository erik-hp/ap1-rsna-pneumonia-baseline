from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .evaluation import box_iou


def patient_labels_from_manifest(
    regions: pd.DataFrame,
    manifest: pd.DataFrame | None,
    group_col: str = "patientId",
    label_col: str = "Target",
) -> dict[str, int]:
    """Retorna um rótulo por paciente para estratificar os folds."""
    groups = regions[group_col].astype(str)

    if manifest is None:
        labels = (
            regions.assign(_group=groups)
            .groupby("_group")[label_col]
            .max()
            .astype(int)
        )
    else:
        if group_col not in manifest.columns or label_col not in manifest.columns:
            raise ValueError(
                f"Manifesto precisa conter '{group_col}' e '{label_col}'."
            )
        labels = (
            manifest.assign(_group=manifest[group_col].astype(str))
            .groupby("_group")[label_col]
            .max()
            .astype(int)
        )

    missing = sorted(set(groups.unique()) - set(labels.index))
    if missing:
        raise ValueError(
            "Há pacientes das regiões ausentes no manifesto: "
            f"{missing[:5]}"
        )

    values = set(labels.loc[groups.unique()].unique())
    if values - {0, 1}:
        raise ValueError("Rótulos de paciente precisam estar em 0/1.")

    return {str(k): int(v) for k, v in labels.items()}


def group_stratified_row_splits(
    groups: pd.Series,
    group_labels: Mapping[str, int],
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cria folds por paciente e devolve índices posicionais das linhas."""
    groups = groups.astype(str).reset_index(drop=True)
    unique_groups = np.asarray(sorted(groups.unique()))
    y_groups = np.asarray([group_labels[str(g)] for g in unique_groups], dtype=int)

    counts = pd.Series(y_groups).value_counts()
    if len(counts) < 2 or int(counts.min()) < n_splits:
        raise ValueError(
            "Poucos pacientes em ao menos uma classe para a divisão "
            f"estratificada em {n_splits} folds: {counts.to_dict()}."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_group_idx, test_group_idx in cv.split(unique_groups, y_groups):
        train_groups = set(unique_groups[train_group_idx])
        test_groups = set(unique_groups[test_group_idx])

        train_rows = np.flatnonzero(groups.isin(train_groups).to_numpy())
        test_rows = np.flatnonzero(groups.isin(test_groups).to_numpy())

        if set(groups.iloc[train_rows]).intersection(set(groups.iloc[test_rows])):
            raise RuntimeError("Vazamento de paciente entre treino e teste.")

        splits.append((train_rows, test_rows))

    return splits


def sample_training_regions(
    frame: pd.DataFrame,
    label_col: str = "Target",
    group_col: str = "patientId",
    max_neg_pos_ratio: float | None = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Mantém todas as regiões positivas e limita negativos apenas no treino.

    Ao menos uma região negativa de cada paciente/grupo é preservada para
    não eliminar pacientes inteiros do CV interno.
    """
    if max_neg_pos_ratio is None:
        return frame.copy()
    if max_neg_pos_ratio <= 0:
        raise ValueError("max_neg_pos_ratio precisa ser positivo ou None.")

    positives = frame[frame[label_col] == 1]
    negatives = frame[frame[label_col] == 0]

    if positives.empty or negatives.empty:
        raise ValueError("Treino regional precisa conter as duas classes.")

    rng = np.random.default_rng(seed)
    mandatory_indices: list[int] = []
    for _, group in negatives.groupby(group_col, sort=True):
        choices = group.index.to_numpy()
        mandatory_indices.append(int(rng.choice(choices)))

    target_negatives = max(
        len(mandatory_indices),
        int(np.ceil(len(positives) * max_neg_pos_ratio)),
    )
    target_negatives = min(target_negatives, len(negatives))

    mandatory = negatives.loc[mandatory_indices]
    remaining = negatives.drop(index=mandatory_indices)
    extra_needed = target_negatives - len(mandatory)

    if extra_needed > 0:
        extra_indices = rng.choice(
            remaining.index.to_numpy(),
            size=extra_needed,
            replace=False,
        )
        selected_negatives = pd.concat(
            [mandatory, remaining.loc[extra_indices]],
            axis=0,
        )
    else:
        selected_negatives = mandatory

    sampled = pd.concat([positives, selected_negatives], axis=0)
    order = rng.permutation(len(sampled))
    return sampled.iloc[order].copy()


def non_max_suppression(
    predictions: pd.DataFrame,
    image_col: str = "patientId",
    iou_threshold: float = 0.30,
    top_k: int | None = 20,
) -> pd.DataFrame:
    """NMS clássico por imagem usando as caixas x,y,width,height."""
    required = {image_col, "x", "y", "width", "height", "score"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predições sem colunas: {sorted(missing)}")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold deve ficar entre 0 e 1.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k precisa ser >= 1 ou None.")

    kept_frames = []

    for _, group in predictions.groupby(image_col, sort=False):
        ordered = group.sort_values("score", ascending=False)
        kept_indices: list[int] = []

        for idx, row in ordered.iterrows():
            box = (row["x"], row["y"], row["width"], row["height"])
            overlaps = [
                box_iou(
                    box,
                    (
                        ordered.loc[kept_idx, "x"],
                        ordered.loc[kept_idx, "y"],
                        ordered.loc[kept_idx, "width"],
                        ordered.loc[kept_idx, "height"],
                    ),
                )
                for kept_idx in kept_indices
            ]
            if overlaps and max(overlaps) > iou_threshold:
                continue

            kept_indices.append(idx)
            if top_k is not None and len(kept_indices) >= top_k:
                break

        if kept_indices:
            kept_frames.append(ordered.loc[kept_indices])

    if not kept_frames:
        return predictions.iloc[0:0].copy()

    return pd.concat(kept_frames, ignore_index=True)
