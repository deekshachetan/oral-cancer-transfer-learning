# Deep transfer learning for oral cancer classification: controlled comparison and external validation

Analysis code for the article *"Deep Transfer Learning for Automated Oral Cancer Classification from Clinical Images: Controlled Comparative Evaluation and External Validation"* (Chetan, Bhat, Navada, Shetty; *Bioengineering*, 2026).

## Contents

| File | Description |
|---|---|
| `notebooks/1_main_analysis_train_evaluate_7cnns.ipynb` | Main analysis: data loading, training of seven ImageNet-pretrained CNNs (seeds 0–2), model selection and locking on validation data, internal and external evaluation, statistics, figures, deep-feature near-duplicate audit, local threshold recalibration. Saved outputs are those of the analysis run of 23 September 2026. |
| `notebooks/1_main_analysis_train_evaluate_7cnns.py` | Plain-code export of the main notebook, including a verification step added during the revision. |
| `notebooks/2_revision_additional_experiments.ipynb` | Additional experiments of the revision: transfer-learning regimes, training duration, augmentation, seeds 3–4 for all models, corrected training-curve figures, augmentation examples. |
| `notebooks/3_revision_additional_analyses.ipynb` | Additional analyses of the revision: paired differences, external pairwise comparisons, calibration and prevalence shift, recalibration over 500 splits, near-duplicate audit with DINOv2 and CLIP, text-overlay masking, class weighting, InceptionV3 at 299 × 299. |

## Order of the analysis

Steps 1–14 of the main notebook were first run on the partition produced by the perceptual-hash audit. A deep-feature audit (Step 15) then identified four images, which were removed, and Steps 1–14 were re-run from the beginning with an unchanged protocol, selection rule and threshold. The model lock was written on 2026-09-23 at 10:21:55 (UTC), and the external cohort was scored at 10:24:46. All reported results come from this repeated analysis. The revision notebooks (November 2026) read these results but never modify them, and were not used to select or change any model. A timeline and the original lock file are provided in Supplementary File S3 of the article.

## Data

The images are **not** included in this repository.

- Development data: five public repositories (Kaggle), cited in the article as references 31–35. The partition file lists are provided in Supplementary File S1.
- External cohort: Piyarathne et al., distributed under restricted access on Zenodo (https://doi.org/10.5281/zenodo.10664056). Request access from its authors. No external images, file names or patient identifiers are included here.

The notebooks expect the folder structure `oral project/oral_clean/` (development images and `split.csv`) and `oral project/external_piyarathne/` (external images and `external.csv`) on Google Drive.

## Trained models

Checkpoints of the seven locked models are archived at Zenodo: [Zenodo DOI of checkpoints].

## Environment

Google Colab with an NVIDIA T4 GPU. Main analysis: Python 3.13, TensorFlow 2.20.0, Keras 3.13.2, NumPy 2.1.3; revision notebooks: TensorFlow 2.21.0. Other packages: pandas, scikit-learn, SciPy, Pillow, Matplotlib; for the audit with independent extractors, PyTorch and Hugging Face `transformers` (DINOv2, CLIP); for text detection, EasyOCR. See `requirements.txt`.

## Citation

If you use this code, please cite the article (citation details will be added on publication) and this archive: [Zenodo DOI of code].

## Licence

Code: MIT licence (see `LICENSE`).
