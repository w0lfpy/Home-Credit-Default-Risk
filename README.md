# Home Credit Default Risk

## Overview

I developed this individual project for the Data Mining & Big Data Analytics module at DBS using Kaggle's [Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk/overview). I investigate whether an applicant will experience payment difficulties and build a machine-learning workflow that covers data understanding, preparation, baseline modelling, advanced modelling, and final evaluation.

My prediction target is `TARGET` (`1`: payment difficulties; `0`: no recorded payment difficulties). Because the positive class is a small minority, I assess ranking performance and class-specific metrics rather than relying on accuracy alone.

### Author

- Jose Suárez Ares

## Project workflow

I run the notebooks in sequence. Notebook 02 creates the processed datasets and split files required by the modelling notebooks. Notebook 03 establishes the baseline; notebook 04 develops and selects the advanced candidate; notebook 05 evaluates that candidate on the held-out test partition and provides interpretation analyses.

| Order | Notebook | What I do | Main outputs |
|---:|---|---|---|
| 1 | [01_data_understanding.ipynb](notebooks/01_data_understanding.ipynb) | I inspect the application training data, variable types, missingness, distributions, and relationships with the target. | Data inventory, variable metadata, summaries, and exploratory figures in `outputs/data_understanding/` and `outputs/figures/`. |
| 2 | [02_data_preparation.ipynb](notebooks/02_data_preparation.ipynb) | I prepare application features, incorporate historical tables, review and select features, fit preprocessing pipelines, and create stratified train, validation, and test partitions. | Processed feature matrices and split files in `data/processed/`; preprocessing artifacts in `models/preprocessing/`; preparation reports in `outputs/data_preparation/`. |
| 3 | [03_baseline_models.ipynb](notebooks/03_baseline_models.ipynb) | I compare baseline classifiers using the prepared baseline matrices and establish a reference against which I assess later models. | Serialized baseline models in `models/baseline/`; benchmark tables and figures in `outputs/baseline_models/`. |
| 4 | [04_advanced_modelling.ipynb](notebooks/04_advanced_modelling.ipynb) | I compare advanced feature configurations, address class imbalance, and tune candidate models. I use validation and cross-validation for model selection, not the held-out test set. | The selected Gradient Boosting candidate in `models/advanced/`; tuning results, feature metadata, and figures in `outputs/advanced_modelling/`. |
| 5 | [05_evaluation.ipynb](notebooks/05_evaluation.ipynb) | I evaluate the selected candidate on the held-out test partition, examine operating thresholds, compare validation and test performance, and interpret features with SHAP. | Evaluation manifests, test metrics, threshold analyses, feature-importance reports, and figures in `outputs/evaluation/`. |

I keep the main notebook workflow separate from [src/test.py](src/test.py), which is an additional standalone pipeline with its own data preparation and train/test split. Its results are not directly comparable to the results from the five-notebook workflow.

## Data and problem definition

I use the Home Credit application dataset, where each applicant is identified by `SK_ID_CURR` and the outcome is stored in `TARGET`. My preparation workflow also uses historical credit and payment records to build additional customer-level predictors. The source files are distributed through the competition's [data page](https://www.kaggle.com/competitions/home-credit-default-risk/data).

To reproduce the complete notebook workflow, I place the following CSV files directly in `data/` and preserve their original names:

- `application_train.csv`
- `HomeCredit_columns_description.csv`
- `bureau.csv` and `bureau_balance.csv`
- `previous_application.csv`
- `POS_CASH_balance.csv`
- `installments_payments.csv`
- `credit_card_balance.csv`

I use `application_test.csv` and `sample_submission.csv` only for additional work with the competition's external test data. My current notebook workflow evaluates an internal held-out test partition; it does not generate a Kaggle submission file automatically.

I have excluded the raw data from version control through [`.gitignore`](.gitignore), so I download it separately when setting up a fresh clone. I also exclude serialized `.joblib` models; I regenerate the required models by running the notebooks. The historical tables contain millions of rows, so I recommend at least 16 GB of RAM for a more comfortable full run. Runtime depends on the machine, particularly for data preparation and hyperparameter tuning.

## Requirements

I pinned the Python package versions in [requirements.txt](requirements.txt). I recommend Python 3.11 or later that is compatible with those pinned packages, together with an isolated virtual environment. From the repository root, I create and activate the environment and install the project dependencies as follows on macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include `ipykernel` and the libraries used by the notebooks. They do not include a Jupyter front end, so I install `notebook` if I want to launch Jupyter Notebook from a terminal:

```bash
python -m pip install notebook
```

On Windows, I create the environment with `python -m venv venv` and activate it with `venv\Scripts\activate` in Command Prompt or PowerShell. In VS Code, I select the `venv` Python interpreter and use that same environment as the notebook kernel. I also need the Python and Jupyter extensions enabled in VS Code.

## Running the notebooks

My notebooks use relative paths such as `../data/` and `../outputs/`. I therefore set the notebook working directory to `notebooks/` before running cells. I run all cells in order and complete each notebook before moving to the next one.

### Option A: Jupyter Notebook

After installing the requirements and the `notebook` front end, I start Jupyter from the repository root with the working directory set to `notebooks/`:

```bash
cd notebooks
jupyter notebook
```

I open and run notebooks 01 through 05 in the order listed above. Starting Jupyter inside `notebooks/` ensures that the relative paths resolve to the repository's sibling folders.

### Option B: VS Code

I open the repository root in VS Code, select the `venv` interpreter and matching notebook kernel, and open the notebook I want to run. Before running its cells, I set the notebook working directory to the notebook's directory (for example, **Change Notebook Working Directory → Notebook's directory**). I then run the notebooks sequentially, completing each stage before starting the next.

## Dependencies between runs

1. I run notebook 01 to generate the data-type metadata and exploratory outputs used by notebook 02.
2. I run notebook 02 to create the processed feature matrices, preprocessing artifacts, and train/validation/test split files.
3. I run notebook 03 to train the baseline models and export benchmark results.
4. I run notebook 04 to tune and save the advanced candidate using the prepared data and baseline references.
5. I run notebook 05 last. It loads the advanced candidate, prepared test matrix, and split targets to produce the final evaluation and SHAP analyses.

If I start from a fresh clone, I ensure that the required Kaggle CSV files are in `data/` before running notebook 01. If I have already generated the processed files and model artifacts, I can resume from the notebook whose inputs are available.

## Metrics and reference results

I use ROC-AUC as my primary model-selection metric and also review PR-AUC, precision, recall, and F1, especially for the positive class. The metrics below are the results recorded in the repository's manifests for the selected advanced Gradient Boosting candidate:

| Evaluation | Metric | Recorded result |
|---|---|---:|
| Validation | ROC-AUC | 0.7783 |
| Validation | PR-AUC | 0.2731 |
| Held-out test | ROC-AUC | 0.7818 |
| Held-out test | PR-AUC | 0.2729 |
| Held-out test, threshold 0.50 | Precision, class 1 | 0.1789 |
| Held-out test, threshold 0.50 | Recall, class 1 | 0.7143 |

I also recorded a threshold of approximately `0.41` that meets a recall requirement of about `0.80`, with lower precision. I use the test partition for final reporting after model selection; it is an internal evaluation split, not Kaggle's external test set or a leaderboard score. I can review the model configuration in [advanced_modelling_manifest.json](outputs/advanced_modelling/results/advanced_modelling_manifest.json) and the final metrics and threshold results in [evaluation_manifest.json](outputs/evaluation/results/evaluation_manifest.json) and [the evaluation results directory](outputs/evaluation/results/).

## Project structure

```text
.
├── data/                 # Kaggle source files and processed datasets
│   └── processed/        # Feature matrices, split files, and configurations
├── models/               # Baseline/advanced estimators and preprocessors
├── notebooks/            # Five sequential analysis and modelling notebooks
├── outputs/              # Reports, manifests, metrics, and figures
├── src/                  # Supporting modules and the optional test.py pipeline
├── requirements.txt      # Pinned Python dependencies
└── README.md
```

I organize preprocessing outputs, trained models, and reports by workflow stage. Because raw data and serialized models are ignored by Git, I recreate or download those artifacts locally as described above.

## Data source and scope

I use data published for Kaggle's [Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk/overview), including the accompanying `HomeCredit_columns_description.csv` data dictionary. I credit Home Credit and Kaggle as the sources of the competition data and problem definition. This repository contains my individual academic analysis and modelling work; I do not present its results as a production credit-decision system.

