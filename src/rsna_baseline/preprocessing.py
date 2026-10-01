import numpy as np
import pydicom
from pydicom.multival import MultiValue
from skimage.exposure import equalize_adapthist
from skimage.transform import resize


SIZE = 256
WIN_SIZES = (48, 64, 96, 128, 160)
WIN_STRIDE = 32
IOU_POS = 0.3


def _dicom_value(value):
    """
    Converte valores DICOM para algo simples de salvar em CSV.
    Alguns campos, como PixelSpacing e WindowCenter, podem possuir
    mais de um valor.
    """
    if value is None:
        return ""

    if isinstance(value, MultiValue):
        return "\\".join(str(v) for v in value)

    return str(value)


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_dicom(path):
    ds = pydicom.dcmread(path)

    img = ds.pixel_array.astype(np.float32)

    photometric = str(
        getattr(ds, "PhotometricInterpretation", "")
    )

    # MONOCHROME1:
    # valores menores representam pixels mais claros.
    # Invertemos para manter o mesmo sentido de intensidade
    # das imagens MONOCHROME2.
    if photometric == "MONOCHROME1":
        img = img.max() - img

    meta = {
        "DICOMPatientID": str(
            getattr(ds, "PatientID", "")
        ),
        "rows": int(
            getattr(ds, "Rows", img.shape[0])
        ),
        "cols": int(
            getattr(ds, "Columns", img.shape[1])
        ),
        "Modality": str(
            getattr(ds, "Modality", "")
        ),
        "ViewPosition": str(
            getattr(ds, "ViewPosition", "")
        ),
        "PhotometricInterpretation": photometric,
        "PixelSpacing": _dicom_value(
            getattr(ds, "PixelSpacing", None)
        ),
        "WindowCenter": _dicom_value(
            getattr(ds, "WindowCenter", None)
        ),
        "WindowWidth": _dicom_value(
            getattr(ds, "WindowWidth", None)
        ),
        "RescaleSlope": _safe_float(
            getattr(ds, "RescaleSlope", 1.0),
            1.0,
        ),
        "RescaleIntercept": _safe_float(
            getattr(ds, "RescaleIntercept", 0.0),
            0.0,
        ),
        "PatientSex": str(
            getattr(ds, "PatientSex", "")
        ),
        "PatientAge": str(
            getattr(ds, "PatientAge", "")
        ),
    }

    return img, meta


def preprocess(img):
    """
    Pipeline de pré-processamento da radiografia:

    1. Resize para 256x256.
    2. Normalização robusta utilizando percentis 1 e 99.
    3. CLAHE para realce de contraste local.
    """

    img = resize(
        img,
        (SIZE, SIZE),
        anti_aliasing=True,
    )

    lo, hi = np.percentile(img, (1, 99))

    img = np.clip(
        (img - lo) / (hi - lo + 1e-8),
        0,
        1,
    )

    img = equalize_adapthist(
        img,
        clip_limit=0.01,
    )

    return img.astype(np.float32)


def windows():
    """
    Gera regiões candidatas quadradas na imagem 256x256.

    Tamanhos: 48, 64, 96, 128 e 160 px.
    Passo: 32 px.

    Contagem por escala: 49 + 49 + 36 + 25 + 16 = 175 regiões.
    """

    for size in WIN_SIZES:
        for y in range(
            0,
            SIZE - size + 1,
            WIN_STRIDE,
        ):
            for x in range(
                0,
                SIZE - size + 1,
                WIN_STRIDE,
            ):
                yield x, y, size, size


def max_iou(box, gts):
    """
    Calcula o maior IoU entre uma região candidata
    e as bounding boxes verdadeiras.
    """

    if len(gts) == 0:
        return 0.0

    ix = np.clip(
        np.minimum(box[2], gts[:, 2])
        - np.maximum(box[0], gts[:, 0]),
        0,
        None,
    )

    iy = np.clip(
        np.minimum(box[3], gts[:, 3])
        - np.maximum(box[1], gts[:, 1]),
        0,
        None,
    )

    intersection = ix * iy

    box_area = (
        (box[2] - box[0])
        * (box[3] - box[1])
    )

    gt_area = (
        (gts[:, 2] - gts[:, 0])
        * (gts[:, 3] - gts[:, 1])
    )

    union = box_area + gt_area - intersection

    iou = intersection / np.maximum(
        union,
        1e-8,
    )

    return float(iou.max())
