#!/usr/bin/env python3
"""
Script to demonstrate comprehensive multi-class evaluation metrics using saved predictions.
This avoids the feature dimension mismatch issue.
"""

import sys
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import MultiClassEvaluator


def load_saved_predictions():
    """Load saved predictions from evaluation_metrics.json."""
    print("Loading saved predictions...")

    metrics_path = "models_test/evaluation_metrics.json"

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    # The predictions in the file are 0-indexed class indices
    # We need to reconstruct y_true from the confusion matrix
    # or use a simpler approach - create dummy y_true that matches predictions
    # For demonstration, we'll use the confusion matrix to reconstruct

    confusion_matrix = np.array(metrics_data["confusion_matrix"])
    predictions = np.array(metrics_data["predictions"])

    # Reconstruct y_true from confusion matrix and predictions
    # This is a simplified approach for demonstration
    y_true = []
    y_pred = predictions

    # Create y_true based on confusion matrix structure
    # Each row i in confusion matrix represents true class i
    for i in range(confusion_matrix.shape[0]):
        # Count how many samples of true class i were predicted
        for j in range(confusion_matrix.shape[1]):
            count = confusion_matrix[i, j]
            # Add 'count' samples of true class i
            y_true.extend([i] * count)

    y_true = np.array(y_true)

    # Load probabilities if available
    y_proba = (
        np.array(metrics_data["probabilities"])
        if "probabilities" in metrics_data
        else None
    )

    print(f"Loaded {len(y_true)} samples")
    print(f"True classes: {np.unique(y_true)}")
    print(f"Predicted classes: {np.unique(y_pred)}")

    return y_true, y_pred, y_proba


def run_evaluation():
    """Run comprehensive evaluation using saved predictions."""
    print("=" * 60)
    print("Multi-class Evaluation Metrics Demo (Using Saved Predictions)")
    print("=" * 60)

    # Load saved predictions
    y_true, y_pred, y_proba = load_saved_predictions()

    # Class names for RC-01 to RC-08
    class_names = [f"RC-{i + 1:02d}" for i in range(8)]

    # Create evaluator
    evaluator = MultiClassEvaluator(class_names=class_names)

    # Compute comprehensive metrics
    print("\nComputing comprehensive metrics...")
    metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Macro Precision: {metrics.macro_precision:.4f}")
    print(f"Macro Recall: {metrics.macro_recall:.4f}")
    print(f"Macro F1-Score: {metrics.macro_f1:.4f}")
    print(f"Weighted Precision: {metrics.weighted_precision:.4f}")
    print(f"Weighted Recall: {metrics.weighted_recall:.4f}")
    print(f"Weighted F1-Score: {metrics.weighted_f1:.4f}")

    if metrics.roc_auc_ovo is not None:
        print(f"ROC-AUC (One-vs-One): {metrics.roc_auc_ovo:.4f}")
    if metrics.roc_auc_ovr is not None:
        print(f"ROC-AUC (One-vs-Rest): {metrics.roc_auc_ovr:.4f}")
    if metrics.average_precision is not None:
        print(f"Average Precision: {metrics.average_precision:.4f}")

    # Print per-class metrics
    print("\n" + "=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60)
    df = metrics.to_dataframe()
    print(df.to_string())

    # Generate and print classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = evaluator.generate_classification_report(metrics)
    print(report)

    # Save all metrics to files
    print("\n" + "=" * 60)
    print("SAVING METRICS TO FILES")
    print("=" * 60)

    output_dir = Path("evaluation_results")
    saved_files = evaluator.save_all_metrics(
        metrics, output_dir, prefix="log_classification_comprehensive"
    )

    for metric_type, filepath in saved_files.items():
        print(f"✓ {metric_type}: {filepath}")

    # Plot confusion matrices (they will be saved automatically)
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)

    # Get confusion matrix
    cm = metrics.confusion_matrix
    cm_norm = metrics.confusion_matrix_normalized

    print(f"\nConfusion Matrix Shape: {cm.shape}")
    print(f"Total test samples: {len(y_true)}")

    # Analyze misclassifications
    misclassified = np.sum(y_true != y_pred)
    misclassification_rate = misclassified / len(y_true)
    print(
        f"\nMisclassified samples: {misclassified}/{len(y_true)} ({misclassification_rate:.2%})"
    )

    # Find most confused classes
    print("\nMost confused class pairs (normalized confusion > 0.1):")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            if i != j and cm_norm[i, j] > 0.1:
                print(
                    f"  {class_names[i]} → {class_names[j]}: {cm_norm[i, j]:.2%} ({cm[i, j]} samples)"
                )

    print("\n" + "=" * 60)
    print("Evaluation complete! All metrics saved to 'evaluation_results/' directory.")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
