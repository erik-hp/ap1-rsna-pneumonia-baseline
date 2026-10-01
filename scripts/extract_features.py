import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from rsna_baseline.features import describe
from rsna_baseline.preprocessing import (
    IOU_POS,
    SIZE,
    load_dicom,
    max_iou,
    preprocess,
    windows,
)


SEED = 42


def select_sample(pat, mode, n_total, n_per_class):
    """
    Seleciona os pacientes de forma reproduzível.

    Modos:

    balanced:
        Seleciona a mesma quantidade de positivos e negativos.
        Útil principalmente para testes rápidos do pipeline.

    natural:
        Mantém aproximadamente a proporção original das classes.
        É mais apropriado para o experimento principal.

    all:
        Utiliza todos os pacientes disponíveis.
    """

    if mode == "all":
        sample = pat.copy()

    elif mode == "balanced":
        parts = []

        for _, group in pat.groupby("Target"):
            n = min(n_per_class, len(group))

            parts.append(
                group.sample(
                    n=n,
                    random_state=SEED,
                )
            )

        sample = pd.concat(
            parts,
            ignore_index=True,
        )

    elif mode == "natural":
        if n_total >= len(pat):
            sample = pat.copy()

        else:
            sample, _ = train_test_split(
                pat,
                train_size=n_total,
                random_state=SEED,
                stratify=pat["Target"],
            )

    else:
        raise ValueError(
            f"Modo de amostragem inválido: {mode}"
        )

    return (
        sample
        .sort_values("patientId")
        .reset_index(drop=True)
    )


def process(pid, target, boxes, dcm_path):
    """
    Processa uma radiografia:

    1. Lê o DICOM.
    2. Valida o PatientID.
    3. Pré-processa a imagem.
    4. Extrai features da imagem inteira.
    5. Gera regiões candidatas.
    6. Calcula IoU das regiões.
    7. Extrai features de cada região.
    """

    # Verifica se o arquivo existe
    if not dcm_path.exists():
        raise FileNotFoundError(
            f"DICOM não encontrado: {dcm_path}"
        )

    # Carrega a imagem e os metadados
    img, meta = load_dicom(dcm_path)

    # Validação do identificador do paciente
    dicom_pid = meta.get(
        "DICOMPatientID",
        "",
    )

    if dicom_pid and dicom_pid != pid:
        raise ValueError(
            f"PatientID inconsistente: "
            f"CSV={pid}, "
            f"DICOM={dicom_pid}, "
            f"arquivo={dcm_path}"
        )

    # Relação entre tamanho original e imagem 256x256
    sx = meta["cols"] / SIZE
    sy = meta["rows"] / SIZE

    # Pré-processamento
    im = preprocess(img)

    # Converte as bounding boxes reais
    # para o sistema de coordenadas 256x256
    gts = np.array(
        [
            [
                x / sx,
                y / sy,
                (x + w) / sx,
                (y + h) / sy,
            ]
            for x, y, w, h in boxes
        ],
        dtype=np.float32,
    ).reshape(-1, 4)

    # =========================================================
    # FEATURES DA IMAGEM COMPLETA
    # =========================================================

    full = {
        "patientId": pid,
        "Target": target,
        **describe(im),
    }

    # =========================================================
    # FEATURES DAS REGIÕES CANDIDATAS
    # =========================================================

    regions = []

    for x, y, w, h in windows():

        # Bounding box candidata na imagem 256x256
        candidate_box = (
            x,
            y,
            x + w,
            y + h,
        )

        # Maior IoU com alguma bounding box verdadeira
        iou = max_iou(
            candidate_box,
            gts,
        )

        # Uma região é positiva caso tenha IoU suficiente
        region_target = int(
            iou >= IOU_POS
        )

        # Coordenadas convertidas novamente
        # para a escala original da radiografia
        region = {
            "patientId": pid,

            "x": round(x * sx),
            "y": round(y * sy),

            "width": round(w * sx),
            "height": round(h * sy),

            "Target": region_target,

            # Mantemos o IoU para auditoria
            "max_iou": iou,

            # Features HOG / LBP / GLCM
            **describe(
                im[
                    y:y + h,
                    x:x + w,
                ]
            ),
        }

        regions.append(region)

    # =========================================================
    # MANIFESTO DA IMAGEM
    # =========================================================

    manifest = {
        "patientId": pid,
        "Target": target,
        "dicom_file": dcm_path.name,
        **meta,
    }

    return manifest, full, regions


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extração de características clássicas "
            "do RSNA Pneumonia Detection Challenge"
        )
    )

    parser.add_argument(
        "--raw",
        default="data/raw",
        help="Diretório contendo os dados brutos.",
    )

    parser.add_argument(
        "--out",
        default="data/processed",
        help="Diretório de saída.",
    )

    parser.add_argument(
        "--sampling",
        choices=[
            "balanced",
            "natural",
            "all",
        ],
        default="natural",
        help=(
            "Estratégia de amostragem: "
            "balanced, natural ou all."
        ),
    )

    parser.add_argument(
        "--n-total",
        type=int,
        default=2000,
        help=(
            "Quantidade total de pacientes "
            "quando sampling=natural."
        ),
    )

    parser.add_argument(
        "--n-per-class",
        type=int,
        default=1000,
        help=(
            "Quantidade máxima por classe "
            "quando sampling=balanced."
        ),
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help=(
            "Número de processos paralelos. "
            "-1 utiliza todos os núcleos disponíveis."
        ),
    )

    args = parser.parse_args()

    raw = Path(args.raw).resolve()
    out = Path(args.out).resolve()
    
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # CARREGAMENTO DOS LABELS
    # =========================================================

    labels_path = (
        raw
        / "stage_2_train_labels.csv"
    )

    images_path = (
        raw
        / "stage_2_train_images"
    )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Arquivo de labels não encontrado: "
            f"{labels_path}"
        )

    if not images_path.exists():
        raise FileNotFoundError(
            f"Diretório das imagens não encontrado: "
            f"{images_path}"
        )

    print(
        f"\nCarregando labels de:\n"
        f"{labels_path}"
    )

    labels = pd.read_csv(
        labels_path
    )

    # =========================================================
    # VALIDAÇÕES BÁSICAS DOS LABELS
    # =========================================================

    required_columns = {
        "patientId",
        "Target",
        "x",
        "y",
        "width",
        "height",
    }

    missing_columns = (
        required_columns
        - set(labels.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colunas ausentes no arquivo de labels: "
            f"{sorted(missing_columns)}"
        )

    invalid_targets = (
        ~labels["Target"].isin([0, 1])
    )

    if invalid_targets.any():
        raise ValueError(
            "Foram encontrados valores de Target "
            "diferentes de 0 e 1."
        )

    # =========================================================
    # UMA LINHA POR PACIENTE
    # =========================================================

    # Pacientes positivos podem possuir várias bounding boxes.
    # Por isso pegamos o maior Target por patientId.
    pat = (
        labels
        .groupby(
            "patientId",
            as_index=False,
        )
        .Target
        .max()
    )

    print("\nDistribuição original:")

    print(
        pat["Target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        f"\nTotal original de pacientes: "
        f"{len(pat)}"
    )

    # =========================================================
    # AMOSTRAGEM
    # =========================================================

    sample = select_sample(
        pat=pat,
        mode=args.sampling,
        n_total=args.n_total,
        n_per_class=args.n_per_class,
    )

    print(
        f"\nEstratégia de amostragem: "
        f"{args.sampling}"
    )

    print(
        "\nDistribuição da amostra:"
    )

    print(
        sample["Target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        f"\nTotal de pacientes selecionados: "
        f"{len(sample)}"
    )

    # Salva exatamente quais pacientes
    # participaram da execução
    sample_manifest_path = (
        out
        / "sample_manifest.csv"
    )

    sample.to_csv(
        sample_manifest_path,
        index=False,
    )

    print(
        f"\nAmostra salva em: "
        f"{sample_manifest_path}"
    )

    # =========================================================
    # BOUNDING BOXES
    # =========================================================

    positive_labels = (
        labels[
            labels["Target"] == 1
        ]
    )

    boxes = {}

    for pid, group in positive_labels.groupby(
        "patientId"
    ):

        boxes[pid] = (
            group[
                [
                    "x",
                    "y",
                    "width",
                    "height",
                ]
            ]
            .values
            .tolist()
        )

    # =========================================================
    # EXTRAÇÃO DAS FEATURES
    # =========================================================

    print(
        "\nIniciando extração de features..."
    )

    results = Parallel(
        n_jobs=args.jobs,
        verbose=5,
    )(
        delayed(process)(
            row.patientId,
            int(row.Target),

            boxes.get(
                row.patientId,
                [],
            ),

            (
                images_path
                / f"{row.patientId}.dcm"
            ),
        )

        for row in sample.itertuples(
            index=False
        )
    )

    # =========================================================
    # SEPARAÇÃO DOS RESULTADOS
    # =========================================================

    manifests = [
        result[0]
        for result in results
    ]

    full_features = [
        result[1]
        for result in results
    ]

    region_features = [
        region
        for result in results
        for region in result[2]
    ]

    # =========================================================
    # SALVA IMAGE MANIFEST
    # =========================================================

    image_manifest = pd.DataFrame(
        manifests
    )

    image_manifest_path = (
        out
        / "image_manifest.csv"
    )

    image_manifest.to_csv(
        image_manifest_path,
        index=False,
    )

    # =========================================================
    # SALVA FEATURES DA IMAGEM COMPLETA
    # =========================================================

    features_all = pd.DataFrame(
        full_features
    )

    features_all_path = (
        out
        / "features_all.csv"
    )

    features_all.to_csv(
        features_all_path,
        index=False,
        float_format="%.6g",
    )

    # =========================================================
    # SALVA FEATURES DAS REGIÕES
    # =========================================================

    features_regions = pd.DataFrame(
        region_features
    )

    features_regions_path = (
        out
        / "features_regions.csv"
    )

    features_regions.to_csv(
        features_regions_path,
        index=False,
        float_format="%.6g",
    )

    # =========================================================
    # SALVA GROUND TRUTH
    # =========================================================

    ground_truth = labels[
        (labels["Target"] == 1)
        &
        (
            labels["patientId"]
            .isin(sample["patientId"])
        )
    ][
        [
            "patientId",
            "x",
            "y",
            "width",
            "height",
        ]
    ].copy()

    ground_truth_path = (
        out
        / "ground_truth_boxes.csv"
    )

    ground_truth.to_csv(
        ground_truth_path,
        index=False,
    )

    # =========================================================
    # RESUMO
    # =========================================================

    print(
        "\n===================================="
    )

    print(
        "EXTRAÇÃO FINALIZADA"
    )

    print(
        "===================================="
    )

    print(
        f"\nPacientes processados: "
        f"{len(sample)}"
    )

    print(
        f"Linhas em features_all.csv: "
        f"{len(features_all)}"
    )

    print(
        f"Linhas em features_regions.csv: "
        f"{len(features_regions)}"
    )

    print(
        f"Bounding boxes verdadeiras: "
        f"{len(ground_truth)}"
    )

    # Contagem das regiões positivas e negativas
    if not features_regions.empty:

        print(
            "\nDistribuição das regiões:"
        )

        print(
            features_regions["Target"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        # Imagens positivas sem nenhuma região positiva
        region_positive_counts = (
            features_regions
            .groupby("patientId")["Target"]
            .sum()
        )

        positive_patients = set(
            sample[
                sample["Target"] == 1
            ]["patientId"]
        )

        patients_without_positive_region = [
            pid
            for pid in positive_patients
            if region_positive_counts.get(
                pid,
                0,
            ) == 0
        ]

        print(
            "\nPacientes positivos sem região "
            f"com IoU >= {IOU_POS}: "
            f"{len(patients_without_positive_region)}"
        )

        if patients_without_positive_region:

            print(
                "ATENÇÃO: existem imagens positivas "
                "que não possuem nenhuma região "
                "candidata positiva."
            )

    print(
        "\nArquivos gerados:"
    )

    print(
        f"  {sample_manifest_path}"
    )

    print(
        f"  {image_manifest_path}"
    )

    print(
        f"  {features_all_path}"
    )

    print(
        f"  {features_regions_path}"
    )

    print(
        f"  {ground_truth_path}"
    )


if __name__ == "__main__":
    main()
