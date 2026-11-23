# Project documentation

## 1. Background

**Academic project:** *Brain Tumor Detection and Survival Prediction Using Deep Learning Techniques*  
**Project authors:** B. Sai Kiran, S. Yoshitha, D. Asharitha, S. Yoshritha  
**Institution:** Sree Vidyanikethan Engineering College (JNTUA affiliated)  
**Academic period:** 2021–2025

The academic project addresses automated MRI brain-tumor segmentation and research into survival prediction. This repository provides an interactive visualization companion that lets users examine volumetric images and segmentation annotations.

## 2. Academic research workflow

The report presents the following conceptual sequence:

1. **Image acquisition:** BraTS 2020 data, with T1, T1ce, T2, and FLAIR imaging modalities.
2. **Preprocessing:** image normalization, cropping and standardization of volumetric data.
3. **Segmentation:** a proposed 3D V-Net architecture for subregion delineation.
4. **Feature extraction:** tumor-region volume and other spatial or image descriptors.
5. **Survival analysis:** a Cox proportional hazards formulation in the methodology. The appended code also presents a Random Forest regressor for survival-day prediction.
6. **Evaluation:** segmentation overlap statistics such as Dice and IoU; appropriate survival-analysis performance measures depend on the implemented estimator and outcome definitions.

The original report and its results belong to the credited research team. The deployed visualization application is not a complete execution of that trained research pipeline.

## 3. Website architecture

```text
NIfTI / NumPy image  ──> Load and validate ──> Select plane, slice and channel
                                                       │
Existing label mask ──> Validate shape ──> Overlay and compute label statistics
                                                       │
                                                 PNG / CSV export
```

### Input requirements

- A 3D numeric volume or a 4D volume with the modality/channel dimension last.
- Optional 3D label mask with dimensions matching the input image's spatial axes.
- NIfTI header voxel spacing where available; NumPy files require a manual spacing value for volumetric measurements.
- MRI and label volumes must be spatially aligned. Matching shapes alone does not guarantee correct clinical registration.

### Current features

| Function | Status |
| --- | --- |
| Interactive multi-planar MRI visualization | Implemented |
| Synthetic, reproducible test image | Implemented |
| Uploaded label-mask overlay | Implemented |
| Annotated-region volume statistics | Implemented |
| PNG and CSV export | Implemented |
| Trained V-Net tumor inference | Not included |
| Fitted survival model inference | Not included |
| Clinical validity testing | Not included |

## 4. Implementation

- `app.py`: Streamlit UI and input selection, MRI slice navigation and downloads.
- `mri_utils.py`: data loading, image display preparation, region metrics and synthetic data generation.
- `requirements.txt`: required Python packages.
- `tests/test_mri_utils.py`: validation and utility tests.

The app processes file uploads in the host environment. It does not intentionally write uploaded MRI files to persistent local storage. Upload only appropriately shareable research data.

## 5. Technical considerations and limitations

- The built-in image is **synthetic**, not a BraTS patient scan.
- Overlay masks are supplied by the user or built into the synthetic example; they are not predicted by a trained model.
- Regional volumes rely on correct voxel dimensions and valid registration.
- The public demo has not been evaluated for diagnosis, clinical decision-making, or survival prognosis.
- Deployment of full 3D segmentation inference requires verified model weights, consistent preprocessing, license-compliant MRI datasets, and compute capacity.

## 6. Future extensions

1. Integrate validated trained segmentation checkpoints.
2. Reproduce training-time image normalization, alignment, label mapping, and cropping.
3. Quantify performance on held-out MRI subjects.
4. Integrate a properly fitted time-to-event survival model if data and model artifacts permit.
5. Publish reproducible model evaluation protocols and limitations.
