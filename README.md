# Brain MRI Explorer
### Brain Tumor Segmentation Research | Interactive MRI Visualization

**Live application:** https://brain-tumor-mri-research.streamlit.app/ 
**Source repository:** https://github.com/saikiranbitragunta83-cse-AI/brain-tumor-mri-explorer

This repository hosts a browser-based MRI visualization and segmentation-analysis dashboard accompanying the academic project **Brain Tumor Detection and Survival Prediction Using Deep Learning Techniques**. The application enables interactive exploration of volumetric MRI data and visualization of provided segmentation labels.

## Academic project team

- **B. Sai Kiran**
- **S. Yoshitha**
- **D. Asharitha**
- **S. Yoshritha**

**Institution:** Sree Vidyanikethan Engineering College, affiliated with Jawaharlal Nehru Technological University Anantapur.  
**Project period:** 2021–2025.

The four names above credit the original academic project; this Streamlit dashboard is a companion implementation. Research findings in the report and features of the deployed dashboard are described separately below.

## Deployed application features

- Load volumetric MRI data in NIfTI (`.nii`, `.nii.gz`) or NumPy (`.npy`) format.
- Navigate **axial**, **coronal**, and **sagittal** image planes.
- Select a slice and inspect MRI image intensity visually.
- Display an existing segmentation label map as a transparent overlay.
- Summarize annotated voxel counts and volumes when spatial spacing is supplied.
- Download selected image and overlay views as PNG files; export region statistics as CSV.
- Explore an automatically generated **synthetic** sample without external patient data.

## Technical stack

| Component | Technology |
| --- | --- |
| Interactive interface | Streamlit |
| Numerical and image processing | NumPy, Pillow |
| NIfTI MRI input | NiBabel |
| Language | Python |
| Hosting | Streamlit Community Cloud |
| Version control | GitHub |

## Research context

The academic report investigates a pipeline involving **BraTS 2020 MRI data**, preprocessing, volumetric tumor segmentation with **V-Net**, extraction of imaging features, and survival modeling. The report discusses Cox proportional hazards modeling; its appendix also contains a Random Forest survival-days regression implementation. More technical detail is available in [Project documentation](docs/PROJECT_DOCUMENTATION.md).

### Current deployment scope

The public application provides **MRI visualization and analysis of existing annotations**. It does **not** automatically detect tumors or calculate patient-specific survival predictions. Trained model checkpoints and clinical validation artifacts are not part of this repository. This separation avoids misrepresenting the research model as implemented in the deployed web application.

**Research-use note:** This software is a demonstration of visualization workflows, not a medical device. Do not submit identifiable patient information or use its output for medical decisions.

## Run locally

Use Python 3.11 or 3.12:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Project layout

```text
.
├── app.py                      # Streamlit interface
├── mri_utils.py                # MRI loading, visualization and metrics
├── requirements.txt           # Python dependencies
├── .streamlit/config.toml      # Interface configuration
├── docs/PROJECT_DOCUMENTATION.md
└── tests/test_mri_utils.py     # Core tests
```

## Testing

```bash
pip install pytest
pytest -q
```

## Data and reproducibility

The included example is programmatically generated synthetic data and contains no patient records. Real MRI data and annotations are not redistributed in this repository. Users are responsible for ensuring that any uploaded data is authorized, de-identified, and suitable for processing by a public cloud service.
