from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rsna_baseline.evaluation import froc, mean_ap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliação de localização RSNA.")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--image-col", default="patientId")
    parser.add_argument(
        "--image-manifest",
        help="CSV contendo todas as imagens; recomendado para FP/imagem correto.",
    )
    parser.add_argument("--output", default="results/detection_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt = pd.read_csv(args.ground_truth)
    pred = pd.read_csv(args.predictions)

    if args.image_manifest:
        manifest = pd.read_csv(args.image_manifest)
        if args.image_col not in manifest.columns:
            raise ValueError(f"Manifesto não contém '{args.image_col}'.")
        image_ids = manifest[args.image_col].astype(str).tolist()
    else:
        # Limitação documentada: negativos sem GT/predição não aparecem nesta união.
        image_ids = sorted(
            set(gt[args.image_col].astype(str)).union(pred[args.image_col].astype(str))
        )

    metrics = {}
    metrics.update(mean_ap(gt, pred, image_col=args.image_col))
    metrics.update(froc(gt, pred, image_ids=image_ids, image_col=args.image_col))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
