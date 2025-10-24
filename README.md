# Pattern-GAM

Code for the "Correcting misinterpretations of additive models", accepted to NeurIPS 2025: https://openreview.net/forum?id=2ClM0g9OFT.


The `./nam/` subfolder is adapted from [lemeln/nam](https://github.com/lemeln/nam).

---

## Conda Setup

Set up the environment using the provided `environment.yml`:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate <env_name>
```

---

## XAI-TRIS Experiments

To generate XAI-TRIS data and save it to `./artifacts/tetris/data/neurips`, run:

```bash
python -m data.generate_data data_config 8x8 neurips
```

- This uses scenarios from `./data/data_config.json`.
- `generate_linesearch.py` creates datasets for SNR-tuning line search, which is then run in `linesearch.py`.
- `visualise_linesearch.ipynb` generates training result plots.
- `xai_tris_train_models.py` trains the final model parameterizations. Ensure the data folder path at the top of each script matches your setup (e.g., `./artifacts/tetris/data/...`).
- `xai_tris_shape_fns.ipynb` shows shape functions and the interaction plots for XAI-TRIS
- `xai_tris_explanations.py` generates global explanations for the notebooks:
    - `xai_tris_qualitative.ipynb`
    - `xai_tris_quantitative_metrics.ipynb`
- These notebooks generate the plots shown in the paper.
- `fni_emd.ipynb` demonstrates explanation styles in the appendix and their metric scores.

---

## COMPAS Recidivism

- `recid.data` is used for experiments.
- `compas_nam.ipynb` trains the NAM and generates shape functions and PatternGAM results.
- `compas_correlation.py` creates correlation plots for the appendix.

Original data and related articles: [propublica/compas-analysis](https://github.com/propublica/compas-analysis).

---

## MIMIC-IV

- `mimic_preprocessing.py` processes raw MIMIC-IV data for the 24-hour mortality task (tested with v2.0).
- `mimic_nam.py` runs the experiments shown in the main text. Reduce epochs or learners for quicker results.

Data source: [MIMIC-IV v2.0](https://physionet.org/content/mimiciv/2.0/) (requires training, data use agreement, and access request).