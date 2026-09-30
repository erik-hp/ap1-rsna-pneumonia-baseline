from __future__ import annotations

import argparse
import json

from rsna_baseline.io import load_feature_table, parse_families
from rsna_baseline.study import build_ablation_families, dataset_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pré-validação da matriz de características.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--group-col", default="patientId")
    parser.add_argument("--label-col", default="Target")
    parser.add_argument("--families", default="HOG=hog_,LBP=lbp_,GLCM=glcm_")
    parser.add_argument("--auto-ablation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_feature_table(args.features)
    families = (
        build_ablation_families(df, args.families)
        if args.auto_ablation
        else parse_families(args.families, df)
    )
    audit = dataset_audit(df, args.group_col, args.label_col, families)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
