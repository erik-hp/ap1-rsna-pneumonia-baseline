#!/usr/bin/env python3
"""
Baixa e organiza automaticamente os dados do
RSNA Pneumonia Detection Challenge usados pelo projeto.

Requisitos:
- aceitar previamente as regras da competição no Kaggle;
- estar autenticado no KaggleHub;
- instalar as dependências do requirements.txt.

Uso normal:
    python scripts/download_data.py

Para solicitar login interativo:
    python scripts/download_data.py --login

Para baixar novamente e substituir os dados de treino:
    python scripts/download_data.py --force
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


COMPETITION = "rsna-pneumonia-detection-challenge"
DEFAULT_OUTPUT = Path("data/raw")


def count_dicoms(images_dir: Path) -> int:
    """Conta os DICOMs diretamente dentro do diretório de treino."""
    return sum(1 for path in images_dir.glob("*.dcm") if path.is_file())


def dataset_ready(output_dir: Path) -> bool:
    """Verifica se os dois elementos necessários ao pipeline já existem."""
    labels = output_dir / "stage_2_train_labels.csv"
    images = output_dir / "stage_2_train_images"

    return (
        labels.is_file()
        and images.is_dir()
        and count_dicoms(images) > 0
    )


def find_one(root: Path, name: str) -> Path | None:
    """Procura um arquivo ou diretório pelo nome dentro de root."""
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def prepare_train_images(download_dir: Path) -> Path:
    """
    Localiza as imagens de treino.

    O Kaggle pode disponibilizar a competição como diretório já extraído
    ou preservar o arquivo stage_2_train_images.zip. Os dois casos são
    tratados aqui.
    """
    images_dir = find_one(download_dir, "stage_2_train_images")

    if images_dir is not None and images_dir.is_dir():
        return images_dir

    images_zip = find_one(download_dir, "stage_2_train_images.zip")

    if images_zip is None:
        raise FileNotFoundError(
            "Não foi possível localizar 'stage_2_train_images' "
            "nem 'stage_2_train_images.zip' após o download."
        )

    extract_dir = download_dir / "stage_2_train_images"

    print(f"Extraindo {images_zip.name}...")

    with zipfile.ZipFile(images_zip, "r") as zf:
        zf.extractall(extract_dir)

    # Alguns ZIPs podem criar uma subpasta com o mesmo nome.
    nested = extract_dir / "stage_2_train_images"
    if nested.is_dir():
        return nested

    return extract_dir


def install_downloaded_train_data(download_dir: Path, output_dir: Path) -> None:
    """Copia apenas labels e imagens de treino para data/raw."""
    labels_src = find_one(download_dir, "stage_2_train_labels.csv")

    if labels_src is None or not labels_src.is_file():
        raise FileNotFoundError(
            "O arquivo stage_2_train_labels.csv não foi encontrado "
            "nos dados baixados."
        )

    images_src = prepare_train_images(download_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    labels_dst = output_dir / "stage_2_train_labels.csv"
    images_dst = output_dir / "stage_2_train_images"

    if labels_dst.exists():
        labels_dst.unlink()

    if images_dst.exists():
        shutil.rmtree(images_dst)

    print("Organizando os dados de treino...")

    shutil.copy2(labels_src, labels_dst)
    shutil.move(str(images_src), str(images_dst))

    dicom_count = count_dicoms(images_dst)

    if dicom_count == 0:
        raise RuntimeError(
            "A pasta stage_2_train_images foi criada, "
            "mas nenhum arquivo .dcm foi encontrado."
        )

    print("\nDados preparados com sucesso.")
    print(f"Labels: {labels_dst.resolve()}")
    print(f"Imagens: {images_dst.resolve()}")
    print(f"DICOMs encontrados: {dicom_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Baixa e organiza automaticamente os dados de treino "
            "do RSNA Pneumonia Detection Challenge."
        )
    )

    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="Diretório final dos dados. Padrão: data/raw",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Baixa novamente e substitui os dados de treino existentes.",
    )

    parser.add_argument(
        "--login",
        action="store_true",
        help="Solicita autenticação interativa do KaggleHub antes do download.",
    )

    args = parser.parse_args()

    output_dir = Path(args.out).resolve()

    if dataset_ready(output_dir) and not args.force:
        print("Os dados de treino já estão disponíveis.")
        print(f"Diretório: {output_dir}")
        print(
            "Use --force somente se realmente quiser baixar "
            "e substituir os dados novamente."
        )
        return

    try:
        import kagglehub
    except ImportError:
        print(
            "Erro: a biblioteca 'kagglehub' não está instalada.\n"
            "Execute:\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.login:
        print("Abrindo autenticação do KaggleHub...")
        kagglehub.login()

    # Não baixamos diretamente para data/raw porque o KaggleHub pode
    # recusar um output_dir já existente e não vazio. O download é feito
    # em um diretório temporário do projeto e, ao final, mantemos apenas
    # os dados de treino usados pelo pipeline.
    temp_dir = output_dir.parent / ".rsna_kaggle_download"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Competição: {COMPETITION}")
    print("Iniciando download pelo KaggleHub...")
    print(
        "Observação: é necessário ter aceitado previamente "
        "as regras da competição no Kaggle."
    )

    try:
        downloaded_path = kagglehub.competition_download(
            COMPETITION,
            output_dir=str(temp_dir),
            force_download=args.force,
        )

        print(f"Download concluído em: {downloaded_path}")

        install_downloaded_train_data(
            download_dir=temp_dir,
            output_dir=output_dir,
        )

    except Exception as exc:
        print(
            "\nNão foi possível concluir o download/preparação.",
            file=sys.stderr,
        )
        print(f"Motivo: {exc}", file=sys.stderr)
        print(
            "\nVerifique se:\n"
            "1. você aceitou as regras da competição no site do Kaggle;\n"
            "2. sua conta está autenticada;\n"
            "3. kagglehub está instalado e atualizado;\n"
            "4. há espaço livre em disco.\n\n"
            "Para tentar autenticação interativa:\n"
            "  python scripts/download_data.py --login",
            file=sys.stderr,
        )
        raise SystemExit(1)

    finally:
        # Após mover os dados de treino, remove test set e arquivos
        # temporários para evitar ocupar espaço desnecessariamente.
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nEstrutura final esperada:")
    print("data/raw/")
    print("├── stage_2_train_labels.csv")
    print("└── stage_2_train_images/")
    print("    ├── <patientId>.dcm")
    print("    └── ...")
    print(
        "\nAgora você pode executar, por exemplo:\n"
        "  python scripts/extract_features.py "
        "--raw data/raw --out data/processed "
        "--sampling natural --n-total 500 --jobs 2"
    )


if __name__ == "__main__":
    main()
