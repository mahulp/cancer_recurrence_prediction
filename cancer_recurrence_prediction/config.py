# cancer_recurrence_prediction/config.py

# Column types for encoding

ORDINAL_MAPS = {
    'Risk': {'Low': 1, 'Intermediate': 2, 'High': 3},
    'Stage': {'I': 1, 'II': 2, 'III': 3, 'IVA': 4, 'IVB': 4},
    'T': {'T1a': 1, 'T1b': 2, 'T2': 3, 'T3a': 4, 'T3b': 5, 'T4a': 6, 'T4b': 7},
    'N': {'N0': 0, 'N1a': 1, 'N1b': 1},
    'Response': {'Excellent': 0, 'Indeterminate': 1, 'Biochemical Incomplete': 2, 'Structural Incomplete': 3}
}

BINARY_MAPS = {
    'Gender': {'F': 0, 'M': 1},
    'Hx Radiotherapy': {'No': 0, 'Yes': 1},
    'Focality': {'Uni-Focal': 0, 'Multi-Focal': 1},
    'M': {'M0': 0, 'M1': 1}
}

TARGET_ENCODED_COLS = ['Pathology', 'Adenopathy']

# The target column
LABEL_COLUMN = 'Recurred'

# Number of features to select
NUM_SELECTED_FEATURES = 6

# Test split
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Classical model names to run in benchmark
CLASSICAL_MODELS = ['logistic', 'svm', 'random_forest']

# Quantum model parameters
VQC_LAYERS = 1
N_EPOCHS = 30
USE_NOISE_IN_TEST = True
NOISE_LEVEL = 0.05