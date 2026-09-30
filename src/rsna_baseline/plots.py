from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def save_oof_figures(y_true, y_score, y_pred, output_dir: str | Path, stem: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(f"ROC — {stem}")
    fig.tight_layout()
    fig.savefig(output_dir / f"roc_{stem}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(f"Precision-Recall — {stem}")
    fig.tight_layout()
    fig.savefig(output_dir / f"pr_{stem}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1], ax=ax)
    ax.set_title(f"Matriz de confusão — {stem}")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_{stem}.png", dpi=180)
    plt.close(fig)
