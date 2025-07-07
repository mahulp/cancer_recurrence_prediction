# cancer_recurrence_prediction/data_preprocessing.py

"""
Data preprocessing module for cancer recurrence prediction.

This module handles:
- Loading raw data
- Train-test splitting
- Encoding categorical features (binary, ordinal, and target encoding)
- Standardization of features
- Feature selection based on Random Forest importance

Encoders are trained only on the training set and reused for new test data
to prevent data leakage. Target encoding is applied safely using training
label information only.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from . import config

def load_data(filepath):
    """
    Load dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(filepath)

def train_test_split_data(df):
    """
    Split the dataset into training and testing sets. Also change the label values to binary (1/0) from Yes/No.

    Args:
        df (pd.DataFrame): Full dataset including label column.

    Returns:
        X_train, X_test, y_train, y_test (pd.DataFrames/Series): Split data.
    """
    X = df.drop(columns=[config.LABEL_COLUMN])
    y = df[config.LABEL_COLUMN]
    if y.dtype==object:
        y = y.map({'No': 0, 'Yes': 1})
    
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE
    )

class Preprocessor:
    """
    Preprocessing class that handles binary encoding, ordinal encoding,
    target encoding, and feature scaling.

    Call `fit()` on training data, then `transform()` on both training and
    test data or future inputs.
    """

    def __init__(self):
        self.target_encoding_maps = {}
        self.target_encoding_defaults = {}
        self.scaler = None

    def fit(self, X, y):
        """
        Fit encoding and scaling transformations using training data.

        Args:
            X (pd.DataFrame): Training features (raw).
            y (pd.Series): Corresponding labels.
        """
        X = X.copy()
        y = y.copy()

        # Binary encoding
        for col, mapping in config.BINARY_MAPS.items():
            if X[col].dtype == object:
                X[col] = X[col].map(mapping)

        # Ordinal encoding
        for col, mapping in config.ORDINAL_MAPS.items():
            if X[col].dtype == object:
                X[col] = X[col].map(mapping)

        # Target encoding: fit median of target grouped by feature
        for col in config.TARGET_ENCODED_COLS:
            medians = y.groupby(X[col]).median()
            self.target_encoding_maps[col] = medians.to_dict()
            self.target_encoding_defaults[col] = medians.median()
            X[col] = X[col].map(self.target_encoding_maps[col])

        # Fit scaler on fully transformed training data
        self.scaler = StandardScaler()
        self.scaler.fit(X)

    def transform(self, X):
        """
        Apply learned transformations to new data (e.g., test set).

        Args:
            X (pd.DataFrame): Data to transform (raw).

        Returns:
            pd.DataFrame: Transformed and scaled features.
        """
        X = X.copy()

        # Binary encoding
        for col, mapping in config.BINARY_MAPS.items():
            if X[col].dtype == object:
                X[col] = X[col].map(mapping)

        # Ordinal encoding
        for col, mapping in config.ORDINAL_MAPS.items():
            if X[col].dtype == object:
                X[col] = X[col].map(mapping)

        # Target encoding using stored maps; fill missing categories
        for col in config.TARGET_ENCODED_COLS:
            X[col] = X[col].map(self.target_encoding_maps[col])
            X[col] = X[col].fillna(self.target_encoding_defaults[col])

        # Scale
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns = X.columns)


def select_important_features(X, y):
    """
    Select top-k features based on Random Forest importance.

    Args:
        X (pd.DataFrame): Input features.
        y (pd.Series): Labels.

    Returns:
        pd.Index: Names of selected top features.
    """
    rf = RandomForestClassifier(class_weight = 'balanced', random_state = config.RANDOM_STATE)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:config.NUM_SELECTED_FEATURES]
    return X.columns[indices]