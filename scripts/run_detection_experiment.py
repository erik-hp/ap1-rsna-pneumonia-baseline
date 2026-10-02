from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from rsna_baseline.detection import (
    group_stratified_row_splits,
    non_max_suppression,
    patient_labels_from_manifest,
    sample_training_regions,
)
from rsna_baseline.evaluation import classification_metrics, froc, mean_ap
from rsna_baseline.io import ensure_columns, load_feature_table
from rsna_baseline.modeling import build_model_specs, continuous_score
from rsna_baseline.study import build_ablation_families, dataset_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecção clássica RSNA com regiões candidatas e OOF por paciente."
    )
    parser.add_argument("--features-regions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--group-col", default="patientId")
    parser.add_argument("--label-col", default="Target")
    parser.add_argument(
        "--families",
        default="HOG=hog_,LBP=lbp_,GLCM=glcm_",
    )
    parser.add_argument(
        "--family",
        default="HOG+LBP+GLCM",
        help="Família/combinação usada no detector regional.",
    )
    parser.add_argument(
        "--model",
        choices=["svm_linear", "svm_rbf", "random_forest", "hist_gradient_boosting"],
        default="hist_gradient_boosting",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="none",
        help="O padrão usa undersampling de negativos; pesos podem ser testados separadamente.",
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--scoring", default="average_precision")
    parser.add_argument("--max-neg-pos-ratio", type=float, default=3.0)
    parser.add_argument("--nms-iou", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_detection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regions = load_feature_table(args.features_regions)
    ensure_columns(
        regions,
        [
            args.group_col,
            args.label_col,
            "x",
            "y",
            "width",
            "height",
        ],
    )

    gt = pd.read_csv(args.ground_truth)
    manifest = pd.read_csv(args.image_manifest)
    ensure_columns(gt, [args.group_col, "x", "y", "width", "height"])
    ensure_columns(manifest, [args.group_col, args.label_col])

    families = build_ablation_families(regions, args.families)
    if args.family not in families:
        raise ValueError(
            f"Família '{args.family}' não existe. Disponíveis: {list(families)}"
        )
    feature_columns = families[args.family]

    audit = dataset_audit(
        regions,
        args.group_col,
        args.label_col,
        {args.family: feature_columns},
    )
    group_labels = patient_labels_from_manifest(
        regions,
        manifest,
        group_col=args.group_col,
        label_col=args.label_col,
    )

    specs = build_model_specs(
        random_state=args.seed,
        class_weight_mode=args.class_weight,
        pca_variance=None,
    )
    spec = specs[args.model]

    outer_splits = group_stratified_row_splits(
        regions[args.group_col],
        group_labels,
        n_splits=args.outer_splits,
        seed=args.seed,
    )

    candidate_frames = []
    metric_rows = []
    best_params = []

    for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        train = regions.iloc[train_idx].copy()
        test = regions.iloc[test_idx].copy()

        train_groups = set(train[args.group_col].astype(str))
        test_groups = set(test[args.group_col].astype(str))
        if train_groups.intersection(test_groups):
            raise RuntimeError(f"Vazamento de paciente no fold {fold}.")

        sampled_train = sample_training_regions(
            train,
            label_col=args.label_col,
            group_col=args.group_col,
            max_neg_pos_ratio=args.max_neg_pos_ratio,
            seed=args.seed + fold,
        ).reset_index(drop=True)

        inner_splits = group_stratified_row_splits(
            sampled_train[args.group_col],
            group_labels,
            n_splits=args.inner_splits,
            seed=args.seed + 100 + fold,
        )

        search = GridSearchCV(
            estimator=spec.estimator,
            param_grid=spec.param_grid,
            scoring=args.scoring,
            cv=inner_splits,
            refit=True,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(
            sampled_train[feature_columns],
            sampled_train[args.label_col].astype(int),
        )

        best = search.best_estimator_
        y_true = test[args.label_col].astype(int).to_numpy()
        y_score = np.asarray(
            continuous_score(best, test[feature_columns]),
            dtype=float,
        )
        y_pred = np.asarray(best.predict(test[feature_columns]), dtype=int)

        metric_rows.append(
            {
                "fold": fold,
                "model": args.model,
                "feature_family": args.family,
                "n_train_regions_raw": int(len(train)),
                "n_train_regions_sampled": int(len(sampled_train)),
                "n_test_regions": int(len(test)),
                "n_train_patients": int(len(train_groups)),
                "n_test_patients": int(len(test_groups)),
                **classification_metrics(y_true, y_score, y_pred),
            }
        )
        best_params.append(
            {
                "fold": fold,
                "best_score_inner": float(search.best_score_),
                "best_params": search.best_params_,
            }
        )

        meta = test[
            [args.group_col, "x", "y", "width", "height", args.label_col]
        ].copy()
        meta = meta.rename(columns={args.label_col: "y_true"})
        meta["y_pred"] = y_pred
        meta["score"] = y_score
        meta["fold"] = fold
        candidate_frames.append(meta)

        print(
            f"Fold {fold}/{args.outer_splits}: "
            f"train={len(sampled_train):,} regiões, "
            f"test={len(test):,}, "
            f"AUC-PR regional={metric_rows[-1]['auc_pr']:.4f}"
        )

    candidates = pd.concat(candidate_frames, ignore_index=True)

    # Preserve the expensive OOF outputs before NMS/mAP/FROC. If any
    # post-processing step fails, the five trained folds are not lost.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "region_metrics_folds.csv",
        index=False,
    )
    candidates.to_csv(
        output_dir / "detection_candidates_oof.csv.gz",
        index=False,
        compression="gzip",
    )
    (output_dir / "detection_best_params.json").write_text(
        json.dumps(best_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "OOF regional preservado. Iniciando NMS, mAP e FROC "
        "com cálculo FROC incremental."
    )

    predictions = non_max_suppression(
        candidates,
        image_col=args.group_col,
        iou_threshold=args.nms_iou,
        top_k=args.top_k,
    )

    detection_predictions = predictions[
        [args.group_col, "x", "y", "width", "height", "score"]
    ].copy()

    image_ids = manifest[args.group_col].astype(str).tolist()
    detection_metrics = {}
    detection_metrics.update(
        mean_ap(gt, detection_predictions, image_col=args.group_col)
    )
    detection_metrics.update(
        froc(
            gt,
            detection_predictions,
            image_ids=image_ids,
            image_col=args.group_col,
        )
    )

    detection_predictions.to_csv(
        output_dir / "detection_predictions.csv",
        index=False,
    )
    (output_dir / "detection_metrics.json").write_text(
        json.dumps(detection_metrics, indent=2),
        encoding="utf-8",
    )
    manifest_out = {
        "seed": args.seed,
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "model": args.model,
        "feature_family": args.family,
        "class_weight": args.class_weight,
        "max_neg_pos_ratio": args.max_neg_pos_ratio,
        "nms_iou": args.nms_iou,
        "top_k": args.top_k,
        "feature_file": Path(args.features_regions).name,
        "dataset_audit": audit,
    }
    (output_dir / "detection_manifest.json").write_text(
        json.dumps(manifest_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDetecção OOF concluída.")
    print(json.dumps(detection_metrics, indent=2))
    print(f"Resultados: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
