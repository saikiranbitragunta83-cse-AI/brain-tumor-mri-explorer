import io
import numpy as np
import pytest
from mri_utils import (
    as_rgb, csv_bytes, load_volume, make_demo, normalize_image,
    region_summary, take_slice, validate_volume,
)


def test_demo_and_stats():
    arr, mask, spacing = make_demo()
    assert arr.shape == mask.shape == (96, 96, 64)
    assert 0 < np.count_nonzero(mask) < mask.size
    info = region_summary(mask, spacing)
    assert info["annotated_voxels"] == np.count_nonzero(mask)
    assert info["annotated_volume_ml"] > 0


def test_views_and_overlay():
    vol, mask, _ = make_demo()
    for orientation in ("Axial", "Coronal", "Sagittal"):
        sl = take_slice(vol, orientation, 25)
        mk = take_slice(mask, orientation, 25)
        rgb = as_rgb(sl, mk)
        assert rgb.shape == (*sl.shape, 3)
        assert rgb.dtype == np.uint8


def test_numpy_upload_roundtrip():
    sample = np.ones((12, 12, 10, 3), dtype=np.float32)
    output = io.BytesIO()
    np.save(output, sample)
    got, spacing = load_volume(output.getvalue(), "example.npy")
    assert got.shape == sample.shape
    assert spacing == (1.0, 1.0, 1.0)


def test_invalid_shape():
    with pytest.raises(ValueError):
        validate_volume(np.zeros((4, 5)))


def test_csv_has_stats():
    encoded = csv_bytes({"annotated_voxels": 34}, "synthetic")
    assert b"annotated_voxels,34" in encoded
