# scripts/predict_on_data.py

"""
Deploy trained quantum models on new clinical data.

- Loads pre-trained model (variational_classifier / quantum_kernel_classifier) and preprocessor
- Applies preprocessing and feature selection
- Runs prediction and prints/saves results
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np

from cancer_recurrence_prediction.quantum_models import (
    VariationalClassifier,
    QuantumKernelClassifier
)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description = "Predict on new test data with a quantum model.")
    parser.add_argument("--model", type = str, required = True, choices = ["variational_classifier", "quantum_kernel_classifier"],
                        help = "Choose model: variational_classifier or quantum_kernel_classifier")
    parser.add_argument("--input", type = str, required = True, help = "Path to input CSV with raw test data")
    parser.add_argument("--output", type = str, default = "results/predictions.csv", help = "Path to save output predictions")
    args  =  parser.parse_args()
    
    # Load input data
    print(f"Loading data from {args.input}")
    df_new = pd.read_csv(args.input)
    X_new_raw = df_new.drop(columns=["Recurred"]) if "Recurred" in df_new.columns else df_new
    
    # Load preprocessor and transform
    print("Loading preprocessor and top features...")
    pre = load_pickle("results/preprocessor.pkl")
    top_features = load_pickle("results/top_features.pkl")
    X_new_proc = pre.transform(X_new_raw)
    X_new_final = X_new_proc[top_features]

    # Reconstruct trained model and make predictions
    print(f"Loading trained model: {args.model}")
    if args.model == "variational_classifier":
        model = VariationalClassifier.load("results/variational_classifier.pkl")
        y_probs, y_preds = model.predict(X_new_final.values, test=True)
    elif args.model == "quantum_kernel_classifier":
        model = QuantumKernelClassifier.load("results/quantum_kernel_classifier.pkl")
        y_probs = model.predict_probs(X_new_final.values)
        y_preds = (y_probs > 0.5).astype(int)

    # Format results
    df_new["Recurrence_prediction"] = y_preds

    # Save
    os.makedirs("results", exist_ok = True)
    df_new.to_csv(args.output, index = False)
    print(f"\ Predictions saved to {args.output}")


if __name__ == "__main__":
    main()