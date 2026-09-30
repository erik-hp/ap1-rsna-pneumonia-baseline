from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera features sintéticas apenas para smoke test.")
    parser.add_argument("--output", default="data/processed/features_smoke.csv")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 20 or args.n % 2:
        raise ValueError("--n deve ser par e >= 20.")

    rng = np.random.default_rng(args.seed)
    target = np.array([0, 1] * (args.n // 2), dtype=int)
    df = pd.DataFrame(
        {
            "patientId": [f"SMOKE_{i:04d}" for i in range(args.n)],
            "Target": target,
        }
    )

    for prefix, shift in (("hog", 0.35), ("lbp", 0.20), ("glcm", 0.28)):
        for i in range(6):
            df[f"{prefix}_{i:02d}"] = rng.normal(loc=target * shift, scale=1.0, size=args.n)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Smoke dataset criado: {output} ({len(df)} linhas)")


if __name__ == "__main__":
    main()
