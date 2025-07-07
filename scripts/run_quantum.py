# scripts/run_quantum.py

"""
Run and save quantum models:
- VariationalClassifier (QNN)
- QuantumKernelClassifier (Kernel-based)

Saves model parameters, metrics and plots.
"""

import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np

from cancer_recurrence_prediction import config
from cancer_recurrence_prediction.data_preprocessing import (
    load_data, train_test_split_data, Preprocessor, select_important_features
)
from cancer_recurrence_prediction.quantum_models import (
    VariationalClassifier, QuantumKernelClassifier
)


def plot_curves(metrics, model_name, save_dir):
    """Plot ROC and PR curves and save as PNG."""
    # ROC
    fpr, tpr, _ = metrics["roc_curve"]
    plt.figure()
    plt.plot(fpr, tpr, label = f"{model_name} ROC (AUC = {np.abs(np.trapz(tpr, fpr)):.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"{model_name}_roc.png"))
    plt.close()

    # PR
    precision, recall, _ = metrics["pr_curve"]
    plt.figure()
    plt.plot(recall, precision, label = f"{model_name} PR (AUC = {np.abs(np.trapz(precision, recall)):.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"{model_name}_pr.png"))
    plt.close()


def save_metrics(metrics, model_name, save_dir):
    """Save metrics as JSON."""
    metrics_copy = {k: v for k, v in metrics.items() if not isinstance(v, tuple)}
    path = os.path.join(save_dir, f"{model_name}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics_copy, f, indent = 4)

def save_pickle(obj, name, save_dir):
    with open(os.path.join(save_dir, f"{name}.pkl"), "wb") as f:
        pickle.dump(obj, f)

def main():
    os.makedirs("results", exist_ok = True)

    # Step 1: Load and split data
    df = load_data("data/medical_data.csv")
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # Step 2: Preprocess and scale
    pre = Preprocessor()
    pre.fit(X_train, y_train)
    X_train_proc = pre.transform(X_train)
    X_test_proc = pre.transform(X_test)
    
    # Save preprocesser
    save_pickle(pre, "preprocessor", "results")

    # Step 3: Feature selection
    top_features = select_important_features(X_train_proc, y_train)
    save_pickle(top_features, "top_features", "results")
    X_train_final = X_train_proc[top_features]
    X_test_final = X_test_proc[top_features]
    
    # Step 4: Train and evaluate Variational Classifier
    vc = VariationalClassifier(
        n_qubits = X_train_final.shape[1],
        layers = config.VQC_LAYERS,
        use_noise_in_test = config.USE_NOISE_IN_TEST,
        noise_level = config.NOISE_LEVEL
    )
    
    print("\nTraining Variational Classifier...")
    vc.fit(X_train_final.values, y_train.values)
    vc_metrics = vc.evaluate(X_test_final.values, y_test.values)

    vc.save("results/variational_classifier.pkl")
    save_metrics(vc_metrics, "variational_classifier", "results")
    plot_curves(vc_metrics, "variational_classifier", "results")

    # Step 5: Train and evaluate Quantum Kernel Classifier
    qk = QuantumKernelClassifier(
        n_qubits = X_train_final.shape[1],
        use_noise_in_test = config.USE_NOISE_IN_TEST,
        noise_level = config.NOISE_LEVEL
    )
    
    print("\nTraining Quantum Kernel Classifier...")
    qk.fit(X_train_final.values, y_train.values)
    qk_metrics = qk.evaluate(X_test_final.values, y_test.values)

    qk.save("results/quantum_kernel_classifier.pkl")
    save_metrics(qk_metrics, "quantum_kernel_classifier", "results")
    plot_curves(qk_metrics, "quantum_kernel_classifier", "results")

    print("\n All quantum models trained and saved to /results.")


if __name__ == "__main__":
    main()