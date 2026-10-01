"""RSNA 2018: DICOM -> pré-processamento -> regiões candidatas -> HOG/LBP/GLCM -> CSVs.
Uso: python scripts/extract_features.py --raw data/raw --out data/processed --n-per-class 1000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rsna_baseline.features import describe
from rsna_baseline.preprocessing import IOU_POS, SIZE, load_dicom, max_iou, preprocess, windows

SEED = 42


def process(pid, target, boxes, dcm_path):
    img, meta = load_dicom(dcm_path)
    sx, sy = meta["cols"] / SIZE, meta["rows"] / SIZE
    im = preprocess(img)
    gts = np.array([[x / sx, y / sy, (x + w) / sx, (y + h) / sy] for x, y, w, h in boxes]).reshape(-1, 4)
    full = {"patientId": pid, "Target": target, **describe(im)}
    regions = []
    for x, y, w, h in windows():
        t = int(max_iou((x, y, x + w, y + h), gts) >= IOU_POS)
        # coordenadas devolvidas na escala original da imagem, comparáveis às caixas reais
        regions.append({"patientId": pid, "x": round(x * sx), "y": round(y * sy),
                        "width": round(w * sx), "height": round(h * sy), "Target": t,
                        **describe(im[y:y + h, x:x + w])})
    return {"patientId": pid, "Target": target, **meta}, full, regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(raw / "stage_2_train_labels.csv")
    pat = labels.groupby("patientId", as_index=False).Target.max()
    sample = pd.concat([g.sample(min(a.n_per_class, len(g)), random_state=SEED) for _, g in pat.groupby("Target")])
    sample = sample.sort_values("patientId").reset_index(drop=True)
    print(sample.Target.value_counts().to_string())

    pos = labels[labels.Target == 1].groupby("patientId")[["x", "y", "width", "height"]]
    boxes = {pid: g.values.tolist() for pid, g in pos}
    res = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(process)(r.patientId, int(r.Target), boxes.get(r.patientId, []),
                         raw / "stage_2_train_images" / f"{r.patientId}.dcm")
        for r in sample.itertuples())

    fmt = dict(index=False, float_format="%.6g")
    pd.DataFrame([r[0] for r in res]).to_csv(out / "image_manifest.csv", index=False)
    pd.DataFrame([r[1] for r in res]).to_csv(out / "features_all.csv", **fmt)
    pd.DataFrame([x for r in res for x in r[2]]).to_csv(out / "features_regions.csv", **fmt)
    gt = labels[(labels.Target == 1) & labels.patientId.isin(sample.patientId)]
    gt[["patientId", "x", "y", "width", "height"]].to_csv(out / "ground_truth_boxes.csv", index=False)


if __name__ == "__main__":
    main()
