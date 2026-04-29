## Contents

- `figure1.py` — calibration regression and gap definitions
- `figure2.py` — sex-specific bias and personalized calibration
- `figure3.py` — prediction-interval stratification
- `figure4.py` — predictability spectrum and mediation b-path identity
- `mediation_analysis.R` — mediation atlas across organ aging models and diseases (produces input for Fig. 4C)

## Requirements

Python 3.9+ with `numpy`, `pandas`, `scipy`, `matplotlib`.
R 4.2+ with `data.table`.

## Data

Analyses use organ age estimates and phenotype data derived from Oh et al., *Nat. Med.* 31, 2703–2711 (2025), based on the UK Biobank. Data are not included in this repository. Approved researchers can access the UK Biobank through https://www.ukbiobank.ac.uk.

Each script expects the following files under `data/`:

- `UKB_aging_model_age_prediction.csv` — predicted ages per individual and tissue, with columns `IID`, `Age`, `Sex`, `pred_Age`, `tissue`, `split`
- `tte_diseases.csv` — time-to-event phenotypes (used for Fig. 3 KM panel)
- `mediation_atlas_results.csv` — output of `mediation_analysis.R` (used for Fig. 4 panel C)
- Disease phenotype files referenced in `mediation_analysis.R`

Update the `DATA_DIR` constant at the top of each script to point to your local data directory.

## Usage

```bash
python figure1.py
python figure2.py
python figure3.py
python figure4.py
Rscript mediation_analysis.R
```

Each script writes `<name>.png` and `<name>.pdf` to the current directory.
