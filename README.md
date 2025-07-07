# Cancer Recurrence Prediction Package

This package contains a hybrid **quantum-classical machine learning pipeline** for predicting cancer recurrence from clinical data. It was developed as a specific case-study.

## Installation

### 1. Create and activate a virtual environment (recommended)

### 2. Install the package in editable mode
From the root project directory (where `setup.py` is located):

```bash
pip install -e .
```


## Modules Overview

| Module                     | Description |
|----------------------------|-------------|
| `data_preprocessing.py`    | Train-test split, encoders (binary, ordinal, target), scaling, and feature selection |
| `classical_models.py`      | Train & benchmark classical models |
| `quantum_models.py`        | Variational classifier and Quantum Kernel classifier, with optional noise |
| `config.py`                | Various settings |

---

## Scripts

### Run all classical models
```bash
python scripts/run_classical.py
```

- Trains all classical models
- Prints metrics and generates ROC & PR plots

### Run quantum models (VQC + kernel)
```bash
python scripts/run_quantum.py
```

- Trains both quantum models
- Evaluates on test data using noise toggle from `config_quantum.py`
- Saves models, metrics, and plots to `results/`



## Predict on New Data

To classify new samples after training:

```bash
python scripts/predict_new.py   --model variational_classifier   --input data/new_patients.csv   --output results/new_predictions.csv
```

Or for kernel-based:
```bash
python scripts/predict_new.py   --model quantum_kernel_classifier   --input data/new_patients.csv
```

- Automatically uses saved preprocessor and feature selection
- Uses noise setting embedded in the trained model
- Outputs predictions and probabilities

