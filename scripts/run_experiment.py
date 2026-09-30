from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rsna_baseline.experiment import ExperimentConfig, run_nested_cv, summarize_metrics
from rsna_baseline.io import ensure_columns, load_feature_table, parse_families
from rsna_baseline.modeling import build_model_specs
from rsna_baseline.plots import save_oof_figures
from rsna_baseline.study import (
    build_ablation_families,
    build_error_cases,
    dataframe_to_markdown,
    dataset_audit,
    experiment_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline clássico RSNA 2018.")
    parser.add_argument("--features", required=True, help="CSV/Parquet com features hand-crafted.")
    parser.add_argument("--group-col", default="patientId")
    parser.add_argument("--label-col", default="Target")
    parser.add_argument(
        "--families",
        default="HOG=hog_,LBP=lbp_,GLCM=glcm_",
        help='Famílias base, ex.: "HOG=hog_,LBP=lbp_,GLCM=glcm_".',
    )
    parser.add_argument(
        "--auto-ablation",
        action="store_true",
        help="Executa todas as combinações não vazias das famílias informadas.",
    )
    parser.add_argument(
        "--models",
        default="svm_linear,svm_rbf,random_forest,hist_gradient_boosting",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Permite ablação do tratamento de desbalanceamento.",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=None,
        help="Opcional: PCA dentro do pipeline preservando a fração de variância informada, ex. 0.95.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--scoring", default="average_precision")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--top-error-cases", type=int, default=10)
    parser.add_argument(
        "--plots",
        choices=["best", "all", "none"],
        default="best",
        help="Por padrão salva ROC/PR/confusão apenas da maior AUC-PR média.",
    )
    return parser.parse_args()


def _plot_selected(
    predictions: pd.DataFrame,
    summary_numeric: pd.DataFrame,
    figures_dir: Path,
    mode: str,
) -> None:
    if mode == "none":
        return

    groups = list(predictions.groupby(["feature_family", "model"]))
    if mode == "best":
        best = summary_numeric.sort_values("auc_pr_mean", ascending=False).iloc[0]
        key = (best["feature_family"], best["model"])
        groups = [(k, frame) for k, frame in groups if k == key]

    for (family, model), frame in groups:
        stem = f"{family}_{model}".lower().replace(" ", "_").replace("+", "_plus_")
        save_oof_figures(
            frame["y_true"],
            frame["y_score"],
            frame["y_pred"],
            figures_dir,
            stem,
        )


def main() -> None:
    args = parse_args()
    df = load_feature_table(args.features)
    ensure_columns(df, [args.group_col, args.label_col])

    families = (
        build_ablation_families(df, args.families)
        if args.auto_ablation
        else parse_families(args.families, df)
    )
    audit = dataset_audit(df, args.group_col, args.label_col, families)

    y = df[args.label_col].astype(int)
    groups = df[args.group_col].astype(str)

    all_specs = build_model_specs(
        random_state=args.seed,
        class_weight_mode=args.class_weight,
        pca_variance=args.pca_variance,
    )
    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in requested if name not in all_specs]
    if unknown:
        raise ValueError(f"Modelos desconhecidos: {unknown}")

    # Baseline trivial sempre entra na comparação.
    selected_names = ["dummy"] + [name for name in requested if name != "dummy"]
    selected_specs = {name: all_specs[name] for name in selected_names}

    config = ExperimentConfig(
        seed=args.seed,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        scoring=args.scoring,
    )

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = experiment_manifest(
        seed=args.seed,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        scoring=args.scoring,
        class_weight_mode=args.class_weight,
        pca_variance=args.pca_variance,
        auto_ablation=args.auto_ablation,
        feature_file=args.features,
        models=selected_names,
        families=families,
    )
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
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

    metadata_cols = [
        c
        for c in [args.group_col, args.label_col, "x", "y", "width", "height"]
        if c in df.columns
    ]
    metadata = df[metadata_cols].copy()
    metadata["row_index"] = metadata.index
    predictions = predictions.merge(metadata, on="row_index", how="left", validate="many_to_one")

    summary_numeric, summary_formatted = summarize_metrics(metrics)
    errors = build_error_cases(predictions, top_per_type=args.top_error_cases)

    metrics.to_csv(output_dir / "metrics_folds.csv", index=False)
    summary_numeric.to_csv(output_dir / "summary_numeric.csv", index=False)
    summary_formatted.to_csv(output_dir / "summary_formatted.csv", index=False)
    predictions.to_csv(output_dir / "predictions_oof.csv", index=False)
    errors.to_csv(output_dir / "error_cases_top.csv", index=False)
    (output_dir / "article_table.md").write_text(
        dataframe_to_markdown(summary_formatted),
        encoding="utf-8",
    )
    (output_dir / "best_params.json").write_text(
        json.dumps(params_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.auto_ablation:
        summary_numeric.to_csv(output_dir / "ablation_summary.csv", index=False)

    _plot_selected(predictions, summary_numeric, figures_dir, args.plots)

    print("\nExperimento concluído sem usar o teste oficial do Kaggle.")
    print(f"Resultados: {output_dir.resolve()}")
    print(f"Figuras: {figures_dir.resolve()}")
    print("\nAuditoria de entrada:")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("\nResumo (média ± desvio-padrão):")
    print(summary_formatted.to_string(index=False))


if __name__ == "__main__":
    main()
