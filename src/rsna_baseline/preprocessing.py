import numpy as np
import pydicom
from skimage.exposure import equalize_adapthist
from skimage.transform import resize

SIZE = 256                     # resize da imagem inteira
WIN_SIZES, WIN_STRIDE = (112, 160), 48
IOU_POS = 0.3                  # região é positiva se IoU máximo com uma caixa real >= IOU_POS


def load_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    meta = dict(rows=img.shape[0], cols=img.shape[1], ViewPosition=str(getattr(ds, "ViewPosition", "")),
                PatientSex=str(getattr(ds, "PatientSex", "")), PatientAge=str(getattr(ds, "PatientAge", "")))
    return img, meta


def preprocess(img):
    img = resize(img, (SIZE, SIZE), anti_aliasing=True)
    lo, hi = np.percentile(img, (1, 99))
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return equalize_adapthist(img, clip_limit=0.01).astype(np.float32)


def windows():
    for s in WIN_SIZES:
        for y in range(0, SIZE - s + 1, WIN_STRIDE):
            for x in range(0, SIZE - s + 1, WIN_STRIDE):
                yield x, y, s, s


def max_iou(box, gts):
    if len(gts) == 0:
        return 0.0
    ix = np.clip(np.minimum(box[2], gts[:, 2]) - np.maximum(box[0], gts[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], gts[:, 3]) - np.maximum(box[1], gts[:, 1]), 0, None)
    inter = ix * iy
    union = (box[2] - box[0]) * (box[3] - box[1]) + (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1]) - inter
    return float((inter / union).max())
