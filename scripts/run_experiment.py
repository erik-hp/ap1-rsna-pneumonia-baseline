from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rsna_baseline.experiment import ExperimentConfig, run_nested_cv, summarize_metrics
from rsna_baseline.io import ensure_columns, load_feature_table, parse_families
from rsna_baseline.modeling import build_model_specs
from rsna_baseline.plots import save_oof_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline clássico RSNA 2018.")
    parser.add_argument("--features", required=True, help="CSV/Parquet com features hand-crafted.")
    parser.add_argument("--group-col", default="patientId")
    parser.add_argument("--label-col", default="Target")
    parser.add_argument(
        "--families",
        default="HOG=hog_,LBP=lbp_,GLCM=glcm_,ALL=*",
        help='Ex.: "HOG=hog_,LBP=lbp_,GLCM=glcm_,ALL=*".',
    )
    parser.add_argument(
        "--models",
        default="svm_linear,svm_rbf,random_forest,hist_gradient_boosting",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_feature_table(args.features)
    ensure_columns(df, [args.group_col, args.label_col])

    if df[args.label_col].isna().any():
        raise ValueError("Target contém valores ausentes.")

    y = df[args.label_col].astype(int)
    groups = df[args.group_col].astype(str)
    families = parse_families(args.families, df)

    all_specs = build_model_specs(args.seed)
    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in requested if name not in all_specs]
    if unknown:
        raise ValueError(f"Modelos desconhecidos: {unknown}")

    # O baseline trivial é sempre executado para cumprir o protocolo.
    selected_names = ["dummy"] + [name for name in requested if name != "dummy"]
    selected_specs = {name: all_specs[name] for name in selected_names}

    config = ExperimentConfig(
        seed=args.seed,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
    )

    metrics_frames = []
    prediction_frames = []
    params_rows = []

    for family_name, columns in families.items():
        metrics, predictions, best_params = run_nested_cv(
            X=df[columns],
            y=y,
            groups=groups,
            model_specs=selected_specs,
            feature_family=family_name,
            config=config,
        )
        metrics_frames.append(metrics)
        prediction_frames.append(predictions)
        params_rows.extend(best_params)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    # Preserva metadados úteis para auditoria e eventual avaliação de detecção.
    metadata_cols = [
        c
        for c in [args.group_col, args.label_col, "x", "y", "width", "height"]
        if c in df.columns
    ]
    metadata = df[metadata_cols].copy()
    metadata["row_index"] = metadata.index
    predictions = predictions.merge(metadata, on="row_index", how="left", validate="many_to_one")

    summary_numeric, summary_formatted = summarize_metrics(metrics)

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(output_dir / "metrics_folds.csv", index=False)
    summary_numeric.to_csv(output_dir / "summary_numeric.csv", index=False)
    summary_formatted.to_csv(output_dir / "summary_formatted.csv", index=False)
    predictions.to_csv(output_dir / "predictions_oof.csv", index=False)
    (output_dir / "best_params.json").write_text(
        json.dumps(params_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for (family, model), frame in predictions.groupby(["feature_family", "model"]):
        stem = f"{family}_{model}".lower().replace(" ", "_")
        save_oof_figures(
            frame["y_true"],
            frame["y_score"],
            frame["y_pred"],
            figures_dir,
            stem,
        )

    print("\nExperimento concluído sem usar o teste oficial do Kaggle.")
    print(f"Resultados: {output_dir.resolve()}")
    print(f"Figuras: {figures_dir.resolve()}")
    print("\nResumo (média ± desvio-padrão):")
    print(summary_formatted.to_string(index=False))


if __name__ == "__main__":
    main()
