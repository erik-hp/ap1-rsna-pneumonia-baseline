from __future__ import annotations

from pathlib import Path

import pandas as pd


NON_FEATURE_COLUMNS = {
    "patientId",
    "patient_id",
    "image_id",
    "Target",
    "target",
    "label",
    "class",
    "x",
    "y",
    "width",
    "height",
    "fold",
    "score",
    "max_iou",
}


def load_feature_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Use uma matriz de características .csv ou .parquet.")


def infer_feature_columns(df: pd.DataFrame, prefix: str = "*") -> list[str]:
    numeric = set(df.select_dtypes(include="number").columns)
    candidates = [c for c in df.columns if c in numeric and c not in NON_FEATURE_COLUMNS]
    if prefix != "*":
        candidates = [c for c in candidates if c.startswith(prefix)]
    if not candidates:
        raise ValueError(f"Nenhuma feature numérica encontrada para prefixo '{prefix}'.")
    return candidates


def parse_families(spec: str, df: pd.DataFrame) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Família inválida: {item}. Use NOME=prefixo.")
        name, prefix = item.split("=", 1)
        families[name.strip()] = infer_feature_columns(df, prefix.strip())
    if not families:
        raise ValueError("Nenhuma família de descritores foi informada.")
    return families


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")
