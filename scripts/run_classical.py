# scripts/run_classical.py

"""
Run the full classical model pipeline:
- Load and preprocess clinical data
- Train/test split
- Benchmark classical models
- Print metrics
- Plot ROC and PR curves
"""

import os
import matplotlib.pyplot as plt

from cancer_recurrence_prediction import config
from cancer_recurrence_prediction.data_preprocessing import load_data, train_test_split_data, Preprocessor
from cancer_recurrence_prediction.classical_models import benchmark_models


def plot_roc_curves(results):
    plt.figure(figsize=(8, 6))
    for model_name, metrics in results.items():
        if "roc_curve" in metrics:
            fpr = metrics["roc_curve"]["fpr"]
            tpr = metrics["roc_curve"]["tpr"]
            auc = metrics["roc_curve"]["auc"]
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_pr_curves(results):
    plt.figure(figsize=(8, 6))
    for model_name, metrics in results.items():
        if "pr_curve" in metrics:
            recall = metrics["pr_curve"]["recall"]
            precision = metrics["pr_curve"]["precision"]
            auc = metrics["pr_curve"]["auc"]
            plt.plot(recall, precision, label=f"{model_name} (AUC = {auc:.2f})")

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    # Load data
    data_path = os.path.join("data", "medical_data.csv")  # change if needed
    df = load_data(data_path)

    # Split
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # Preprocess
    pre = Preprocessor()
    pre.fit(X_train, y_train)
    X_train_proc = pre.transform(X_train)
    X_test_proc = pre.transform(X_test)

    # Benchmark classical models
    results = benchmark_models(X_train_proc, y_train, X_test_proc, y_test, generate_curve_data = True)

    # Print summary
    for model, metrics in results.items():
        print(f"\n{model.upper()}: \n")
        if "error" in metrics:
            print("Error:", metrics["error"])
        else:
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1 Score: {metrics['f1']:.3f}")

    # Plot curves
    plot_roc_curves(results)
    plot_pr_curves(results)


if __name__ == "__main__":
    main()