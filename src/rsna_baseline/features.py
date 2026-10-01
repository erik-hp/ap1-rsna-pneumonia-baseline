import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.transform import resize

CROP = 64                      # resize de cada região antes das features
HOG_P = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
LBP_P = dict(P=8, R=1, method="uniform")
GLCM_D, GLCM_LEVELS = (1, 3), 32
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def describe(region):
    c = resize(region, (CROP, CROP), anti_aliasing=True)
    f = {f"hog_{i}": v for i, v in enumerate(hog(c, **HOG_P))}
    lbp = local_binary_pattern((c * 255).astype(np.uint8), **LBP_P).astype(int)
    hist = np.bincount(lbp.ravel(), minlength=LBP_P["P"] + 2) / lbp.size
    f.update({f"lbp_{i}": v for i, v in enumerate(hist)})
    q = np.minimum((c * GLCM_LEVELS).astype(np.uint8), GLCM_LEVELS - 1)
    g = graycomatrix(q, GLCM_D, [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], GLCM_LEVELS, symmetric=True, normed=True)
    for p in GLCM_PROPS:
        for d, v in zip(GLCM_D, graycoprops(g, p).mean(axis=1)):
            f[f"glcm_{p}_d{d}"] = v
    return {k: float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in f.items()}
