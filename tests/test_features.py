import numpy as np
import pytest

from rsna_baseline.features import describe
from rsna_baseline.preprocessing import IOU_POS, SIZE, max_iou, preprocess, windows

N_FEATURES = 324 + 10 + 12     # HOG + LBP + GLCM


@pytest.fixture
def region():
    return np.random.default_rng(0).random((112, 112)).astype(np.float32)


def test_describe_columns_and_prefixes(region):
    f = describe(region)
    assert len(f) == N_FEATURES
    assert {k.split("_")[0] for k in f} == {"hog", "lbp", "glcm"}
    assert sum(k.startswith("hog_") for k in f) == 324
    assert sum(k.startswith("lbp_") for k in f) == 10
    assert sum(k.startswith("glcm_") for k in f) == 12


def test_describe_is_finite_and_deterministic(region):
    a, b = describe(region), describe(region)
    assert a == b
    assert np.isfinite(list(a.values())).all()


def test_describe_flat_region_has_no_nan():
    assert np.isfinite(list(describe(np.full((112, 112), 0.5, dtype=np.float32)).values())).all()


def test_lbp_histogram_sums_to_one(region):
    f = describe(region)
    assert sum(v for k, v in f.items() if k.startswith("lbp_")) == pytest.approx(1.0)


def test_preprocess_shape_and_range():
    out = preprocess(np.random.default_rng(1).random((300, 400)).astype(np.float32) * 4000)
    assert out.shape == (SIZE, SIZE) and out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1


def test_windows_count_and_bounds():
    ws = list(windows())
    assert len(ws) == 25
    assert all(0 <= x and 0 <= y and x + w <= SIZE and y + h <= SIZE for x, y, w, h in ws)


def test_max_iou():
    gts = np.array([[0, 0, 100, 100]], dtype=float)
    assert max_iou((0, 0, 100, 100), gts) == pytest.approx(1.0)
    assert max_iou((200, 200, 250, 250), gts) == 0.0
    assert max_iou((0, 0, 100, 100), np.empty((0, 4))) == 0.0
    assert max_iou((0, 0, 50, 100), gts) == pytest.approx(0.5) and 0.5 >= IOU_POS
