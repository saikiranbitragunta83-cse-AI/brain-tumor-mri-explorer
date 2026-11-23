"""Small, testable MRI visualization utilities. Not a medical prediction model."""
from __future__ import annotations

import csv
import gzip
import io
from typing import Iterable

import numpy as np
from PIL import Image

MAX_UPLOAD_BYTES = 60 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 240 * 1024 * 1024
MAX_VOXELS = 50_000_000


def validate_volume(volume: np.ndarray) -> np.ndarray:
    """Accept 3-D MRI arrays or 4-D arrays with 1–4 modalities as final axis."""
    result = np.asarray(volume)
    if result.ndim not in (3, 4):
        raise ValueError("Expected a 3-D MRI volume or 4-D MRI volume with modalities on the last axis.")
    if any(int(v) < 8 for v in result.shape[:3]):
        raise ValueError("The three spatial dimensions must each be at least eight voxels.")
    if result.ndim == 4 and not (1 <= result.shape[3] <= 4):
        raise ValueError("4-D arrays must contain one to four modalities on the last axis.")
    if result.size > MAX_VOXELS:
        raise ValueError("This MRI is too large for the free demo. Use a smaller, de-identified sample.")
    if not np.issubdtype(result.dtype, np.number):
        raise ValueError("MRI data must be numeric.")
    return result


def load_volume(file_bytes: bytes, filename: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load .npy/.nii/.nii.gz, keeping everything in memory."""
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("File is above the 60 MiB demo upload limit.")
    lowered = filename.lower()
    if lowered.endswith(".npy"):
        data = np.load(io.BytesIO(file_bytes), allow_pickle=False)
        spacing = (1.0, 1.0, 1.0)  # Unknown for .npy; user may override.
    elif lowered.endswith(".nii") or lowered.endswith(".nii.gz"):
        import nibabel as nib
        if lowered.endswith(".gz"):
            with gzip.GzipFile(fileobj=io.BytesIO(file_bytes)) as gz:
                file_bytes = gz.read(MAX_UNCOMPRESSED_BYTES + 1)
            if len(file_bytes) > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Compressed MRI expands beyond the demo's memory limit.")
        nii = nib.Nifti1Image.from_bytes(file_bytes)
        shape = nii.shape
        if len(shape) not in (3, 4) or np.prod(shape) > MAX_VOXELS:
            raise ValueError("This MRI has unsupported dimensions or exceeds the demo size limit.")
        data = nii.get_fdata(dtype=np.float32)
        spacing = tuple(float(x) for x in nii.header.get_zooms()[:3])
    else:
        raise ValueError("Supported file types: .npy, .nii, .nii.gz")
    data = validate_volume(data)
    if any(not np.isfinite(x) or x <= 0 for x in spacing):
        spacing = (1.0, 1.0, 1.0)
    return data, spacing


def make_demo() -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Artificial, reproducible brain-shaped sample. NOT a real MRI or AI output."""
    shape = (96, 96, 64)
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    brain = ((x - 48) / 39.0) ** 2 + ((y - 48) / 35.0) ** 2 + ((z - 32) / 28.0) ** 2 <= 1
    inner = ((x - 48) / 27.0) ** 2 + ((y - 48) / 25.0) ** 2 + ((z - 32) / 22.0) ** 2 <= 1
    bright = ((x - 60) / 14.0) ** 2 + ((y - 40) / 11.0) ** 2 + ((z - 36) / 13.0) ** 2 <= 1
    core = ((x - 60) / 8.0) ** 2 + ((y - 40) / 7.0) ** 2 + ((z - 36) / 9.0) ** 2 <= 1
    rng = np.random.default_rng(2026)
    noise = rng.normal(0, 9, size=shape).astype(np.float32)
    volume = np.zeros(shape, dtype=np.float32)
    volume[brain] = 68
    volume[inner] = 104
    volume[bright] = 174
    volume[core] = 205
    volume[brain] += noise[brain]
    volume = np.clip(volume, 0, 255)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[bright] = 2
    mask[core] = 1
    return volume, mask, (1.0, 1.0, 1.0)


def take_slice(volume: np.ndarray, axis: str, index: int) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError("take_slice expects one 3-D modality.")
    axis_num = {"Axial": 2, "Coronal": 1, "Sagittal": 0}[axis]
    if not (0 <= index < volume.shape[axis_num]):
        raise IndexError("Slice index out of range.")
    return np.rot90(np.take(volume, index, axis=axis_num))


def normalize_image(img: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(img, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = clean[np.isfinite(clean)]
    if valid.size == 0:
        return np.zeros(clean.shape, dtype=np.uint8)
    low, high = np.percentile(valid, [1, 99])
    if high <= low:
        return np.zeros(clean.shape, dtype=np.uint8)
    return np.clip((clean - low) / (high - low) * 255, 0, 255).astype(np.uint8)


def as_rgb(image: np.ndarray, mask: np.ndarray | None = None, alpha: float = 0.40) -> np.ndarray:
    gray = normalize_image(image)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    if mask is None:
        return rgb.astype(np.uint8)
    if mask.shape != image.shape:
        raise ValueError("Overlay mask and image slices must have identical shapes.")
    colors = {1: (245, 76, 88), 2: (245, 174, 60), 3: (58, 194, 178), 4: (58, 194, 178)}
    for label in np.unique(mask):
        if label == 0:
            continue
        region = mask == label
        color = colors.get(int(label), (194, 97, 235))
        rgb[region] = rgb[region] * (1 - alpha) + np.asarray(color) * alpha
    return rgb.astype(np.uint8)


def png_bytes(image: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def region_summary(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> dict[str, float | int]:
    voxels = int(np.count_nonzero(mask))
    voxel_volume_mm3 = float(np.prod(spacing_mm))
    return {
        "annotated_voxels": voxels,
        "total_voxels": int(mask.size),
        "annotated_percent": 100.0 * voxels / mask.size,
        "annotated_volume_ml": voxels * voxel_volume_mm3 / 1000.0,
    }


def csv_bytes(summary: dict[str, float | int], data_source: str) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["metric", "value"])
    writer.writerow(["data_source", data_source])
    for key, value in summary.items():
        writer.writerow([key, value])
    return buffer.getvalue().encode("utf-8")
