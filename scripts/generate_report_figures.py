from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera figuras consolidadas para o relatório a partir dos resultados já calculados."
    )
    parser.add_argument("--classification-results", required=True)
    parser.add_argument("--detection-results", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} não contém as colunas necessárias: {missing}")


def plot_top_pr_auc(summary: pd.DataFrame, output_path: Path) -> None:
    _require_columns(
        summary,
        ["feature_family", "model", "auc_pr_mean"],
        output_path,
    )

    non_dummy = summary[summary["model"] != "dummy"].copy()
    top = non_dummy.sort_values("auc_pr_mean", ascending=False).head(10)
    labels = top["feature_family"] + " | " + top["model"]

    dummy = summary[summary["model"] == "dummy"]["auc_pr_mean"]
    baseline = float(dummy.mean()) if not dummy.empty else None

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(labels.iloc[::-1], top["auc_pr_mean"].iloc[::-1])
    if baseline is not None:
        ax.axvline(
            baseline,
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline trivial ({baseline:.3f})",
        )
        ax.legend(loc="lower right")
    ax.set_xlabel("PR-AUC média")
    ax.set_ylabel("Descritor + modelo")
    ax.set_title("Top 10 combinações por PR-AUC — classificação")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_specificity(summary: pd.DataFrame, output_path: Path) -> None:
    _require_columns(
        summary,
        [
            "feature_family",
            "model",
            "sensitivity_mean",
            "specificity_mean",
        ],
        output_path,
    )

    frame = summary[summary["model"] != "dummy"].copy()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(frame["specificity_mean"], frame["sensitivity_mean"])

    # Rótulos curtos preservam legibilidade sem esconder pontos.
    for row in frame.itertuples(index=False):
        label = f"{row.feature_family}\n{row.model}"
        ax.annotate(
            label,
            (row.specificity_mean, row.sensitivity_mean),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
        )

    ax.set_xlabel("Especificidade média")
    ax.set_ylabel("Sensibilidade média")
    ax.set_title("Trade-off sensibilidade × especificidade")
    ax.set_xlim(0.55, 1.01)
    ax.set_ylim(0.0, 0.75)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _extract_froc_points(metrics: dict[str, float]) -> pd.DataFrame:
    pattern = re.compile(r"^froc_sens_at_([0-9.]+)_fp_per_image$")
    rows: list[tuple[float, float]] = []

    for key, value in metrics.items():
        match = pattern.match(key)
        if match:
            rows.append((float(match.group(1)), float(value)))

    if not rows:
        raise ValueError("detection_metrics.json não contém pontos FROC.")

    return pd.DataFrame(rows, columns=["fp_per_image", "sensitivity"]).sort_values(
        "fp_per_image"
    )


def plot_froc(metrics: dict[str, float], output_path: Path) -> None:
    froc = _extract_froc_points(metrics)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        froc["fp_per_image"],
        froc["sensitivity"],
        marker="o",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(
        froc["fp_per_image"],
        [f"{value:g}" for value in froc["fp_per_image"]],
    )
    ax.set_xlabel("Falsos positivos por imagem")
    ax.set_ylabel("Sensibilidade")
    ax.set_title("Curva FROC — detecção/localização")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    classification_dir = Path(args.classification_results)
    detection_dir = Path(args.detection_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = classification_dir / "summary_numeric.csv"
    detection_path = detection_dir / "detection_metrics.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Resultado de classificação não encontrado: {summary_path}")
    if not detection_path.exists():
        raise FileNotFoundError(f"Resultado de detecção não encontrado: {detection_path}")

    summary = pd.read_csv(summary_path)
    detection_metrics = json.loads(detection_path.read_text(encoding="utf-8"))

    outputs = [
        output_dir / "classificacao_top10_prauc.png",
        output_dir / "tradeoff_sensibilidade_especificidade.png",
        output_dir / "froc_deteccao.png",
    ]

    plot_top_pr_auc(summary, outputs[0])
    plot_sensitivity_specificity(summary, outputs[1])
    plot_froc(detection_metrics, outputs[2])

    print("\nFiguras para o relatório geradas sem recalcular os experimentos:")
    for output in outputs:
        print(f"  {output.resolve()}")


if __name__ == "__main__":
    main()
