from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_score, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    try:
        auc_roc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc_roc = np.nan

    try:
        auc_pr = average_precision_score(y_true, y_score)
    except ValueError:
        auc_pr = np.nan

    return {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    """IoU para caixas no formato x, y, width, height."""
    ax, ay, aw, ah = map(float, box_a)
    bx, by, bw, bh = map(float, box_b)

    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else 0.0


def _prepare_boxes(df: pd.DataFrame, image_col: str) -> dict[str, list[tuple[float, float, float, float]]]:
    boxes: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    if df.empty:
        return boxes

    required = {image_col, "x", "y", "width", "height"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para caixas: {sorted(missing)}")

    for row in df.itertuples(index=False):
        record = row._asdict()
        boxes[str(record[image_col])].append(
            (
                float(record["x"]),
                float(record["y"]),
                float(record["width"]),
                float(record["height"]),
            )
        )
    return boxes


def detection_ap(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    image_col: str = "patientId",
    iou_threshold: float = 0.5,
) -> float:
    """Average Precision global para uma classe, com matching 1:1 por imagem."""
    if "score" not in predictions.columns:
        raise ValueError("Predições de detecção precisam da coluna 'score'.")

    gt_boxes = _prepare_boxes(ground_truth, image_col)
    total_gt = sum(len(v) for v in gt_boxes.values())
    if total_gt == 0:
        return float("nan")

    matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_boxes.items()}
    preds = predictions.sort_values("score", ascending=False).reset_index(drop=True)

    tp = np.zeros(len(preds), dtype=float)
    fp = np.zeros(len(preds), dtype=float)

    for i, row in preds.iterrows():
        image_id = str(row[image_col])
        pred_box = (row["x"], row["y"], row["width"], row["height"])
        candidates = gt_boxes.get(image_id, [])

        if not candidates:
            fp[i] = 1.0
            continue

        ious = np.asarray([box_iou(pred_box, gt) for gt in candidates])
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold and not matched[image_id][best]:
            tp[i] = 1.0
            matched[image_id][best] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))


def mean_ap(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    image_col: str = "patientId",
    thresholds: Iterable[float] | None = None,
) -> dict[str, float]:
    thresholds = list(thresholds or np.arange(0.50, 0.96, 0.05))
    values = {
        f"ap_iou_{threshold:.2f}": detection_ap(
            ground_truth, predictions, image_col=image_col, iou_threshold=float(threshold)
        )
        for threshold in thresholds
    }
    values["map_50_95"] = float(np.nanmean(list(values.values())))
    return values


def _match_counts(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    image_col: str,
    iou_threshold: float,
) -> tuple[int, int]:
    gt_boxes = _prepare_boxes(ground_truth, image_col)
    matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_boxes.items()}
    tp = 0
    fp = 0

    for _, row in predictions.sort_values("score", ascending=False).iterrows():
        image_id = str(row[image_col])
        pred_box = (row["x"], row["y"], row["width"], row["height"])
        candidates = gt_boxes.get(image_id, [])
        if not candidates:
            fp += 1
            continue

        ious = np.asarray([box_iou(pred_box, gt) for gt in candidates])
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold and not matched[image_id][best]:
            matched[image_id][best] = True
            tp += 1
        else:
            fp += 1
    return tp, fp


def froc(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    image_ids: Iterable[str],
    image_col: str = "patientId",
    iou_threshold: float = 0.5,
    fp_per_image_points: Iterable[float] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
) -> dict[str, float]:
    """
    Calcula FROC em uma única passagem sobre as predições ordenadas.

    A implementação anterior recalculava todo o matching para cada score
    distinto, levando a custo aproximadamente quadrático. Aqui o matching
    1:1 é atualizado incrementalmente e um ponto operacional é registrado
    somente após consumir todas as predições com o mesmo score, preservando
    a semântica de um limiar de score.
    """
    image_ids = list(dict.fromkeys(map(str, image_ids)))
    if not image_ids:
        raise ValueError("FROC requer ao menos um image_id.")

    total_gt = len(ground_truth)
    if total_gt == 0:
        return {
            f"froc_sens_at_{p:g}_fp_per_image": float("nan")
            for p in fp_per_image_points
        }

    gt_boxes = _prepare_boxes(ground_truth, image_col)
    matched = {
        image_id: np.zeros(len(boxes), dtype=bool)
        for image_id, boxes in gt_boxes.items()
    }

    if predictions.empty:
        operating_points = [(0.0, 0.0)]
    else:
        required = {image_col, "x", "y", "width", "height", "score"}
        missing = required.difference(predictions.columns)
        if missing:
            raise ValueError(f"Predições sem colunas para FROC: {sorted(missing)}")

        ordered = predictions.sort_values(
            "score",
            ascending=False,
            kind="stable",
        ).reset_index(drop=True)

        tp = 0
        fp = 0
        operating_points = [(0.0, 0.0)]

        # Agrupar empates é importante: um threshold inclui todas as
        # predições com o mesmo score, nunca apenas parte delas.
        for _, score_group in ordered.groupby("score", sort=False):
            for row in score_group.itertuples(index=False):
                record = row._asdict()
                image_id = str(record[image_col])
                pred_box = (
                    record["x"],
                    record["y"],
                    record["width"],
                    record["height"],
                )
                candidates = gt_boxes.get(image_id, [])

                if not candidates:
                    fp += 1
                    continue

                ious = np.asarray([box_iou(pred_box, gt) for gt in candidates])
                best = int(np.argmax(ious))
                if (
                    ious[best] >= iou_threshold
                    and not matched[image_id][best]
                ):
                    matched[image_id][best] = True
                    tp += 1
                else:
                    fp += 1

            operating_points.append(
                (fp / len(image_ids), tp / total_gt)
            )

    result = {}
    for target in fp_per_image_points:
        sensitivities = [
            sens for fppi, sens in operating_points if fppi <= target
        ]
        result[f"froc_sens_at_{target:g}_fp_per_image"] = float(
            max(sensitivities, default=0.0)
        )
    return result
