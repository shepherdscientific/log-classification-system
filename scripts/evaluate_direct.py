#!/usr/bin/env python3
"""
Script to demonstrate comprehensive multi-class evaluation metrics
by directly using the saved classification report and confusion matrix.
"""

import sys
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import MultiClassEvaluator


def create_metrics_from_saved():
    """Create MultiClassMetrics from saved evaluation data."""
    print("Loading saved evaluation metrics...")

    metrics_path = "models_test/evaluation_metrics.json"

    with open(metrics_path, "r") as f:
        saved_data = json.load(f)

    # Extract data from saved file
    accuracy = saved_data["accuracy"]
    confusion_matrix = np.array(saved_data["confusion_matrix"])

    # Get classification report data
    report = saved_data["classification_report"]

    # Class names for RC-01 to RC-08
    class_names = [f"RC-{i + 1:02d}" for i in range(8)]

    # Extract per-class metrics
    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    per_class_support = {}

    for class_name in class_names:
        if class_name in report:
            per_class_precision[class_name] = report[class_name]["precision"]
            per_class_recall[class_name] = report[class_name]["recall"]
            per_class_f1[class_name] = report[class_name]["f1-score"]
            per_class_support[class_name] = int(report[class_name]["support"])

    # Get macro and weighted averages
    macro_avg = report["macro avg"]
    weighted_avg = report["weighted avg"]

    # Create normalized confusion matrix
    cm_normalized = (
        confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
    )
    cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)

    # Create MultiClassMetrics object
    from src.evaluation.metrics import MultiClassMetrics

    metrics = MultiClassMetrics(
        accuracy=float(accuracy),
        macro_precision=float(macro_avg["precision"]),
        macro_recall=float(macro_avg["recall"]),
        macro_f1=float(macro_avg["f1-score"]),
        weighted_precision=float(weighted_avg["precision"]),
        weighted_recall=float(weighted_avg["recall"]),
        weighted_f1=float(weighted_avg["f1-score"]),
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        per_class_support=per_class_support,
        confusion_matrix=confusion_matrix,
        confusion_matrix_normalized=cm_normalized,
        roc_auc_ovo=None,  # Not available in saved data
        roc_auc_ovr=None,  # Not available in saved data
        average_precision=None,  # Not available in saved data
    )

    print(f"Loaded metrics with accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_avg['f1-score']:.4f}")

    return metrics, class_names


def run_evaluation():
    """Run comprehensive evaluation using saved metrics."""
    print("=" * 60)
    print("Multi-class Evaluation Metrics Demo (Direct from Saved)")
    print("=" * 60)

    # Load saved metrics
    metrics, class_names = create_metrics_from_saved()

    # Create evaluator
    evaluator = MultiClassEvaluator(class_names=class_names)

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

    # Since we don't have y_true and y_pred, create a simplified report
    report_text = f"""
Classification Report (From Saved Metrics)
==========================================

Accuracy: {metrics.accuracy:.4f}
Macro Precision: {metrics.macro_precision:.4f}
Macro Recall: {metrics.macro_recall:.4f}
Macro F1-Score: {metrics.macro_f1:.4f}
Weighted Precision: {metrics.weighted_precision:.4f}
Weighted Recall: {metrics.weighted_recall:.4f}
Weighted F1-Score: {metrics.weighted_f1:.4f}

Per-Class Metrics:
------------------
{df.to_string()}

Confusion Matrix Shape: {metrics.confusion_matrix.shape}
Total Samples: {sum(metrics.per_class_support.values())}
"""
    print(report_text)

    # Save all metrics to files
    print("\n" + "=" * 60)
    print("SAVING METRICS TO FILES")
    print("=" * 60)

    output_dir = Path("evaluation_results")
    saved_files = evaluator.save_all_metrics(
        metrics, output_dir, prefix="log_classification_final"
    )

    # Note: We can't generate the full classification report without y_true/y_pred
    # But we can still save the metrics and confusion matrices
    for metric_type, filepath in saved_files.items():
        if (
            metric_type != "classification_report"
        ):  # Skip report since we don't have y_true/y_pred
            print(f"✓ {metric_type}: {filepath}")

    # Plot confusion matrices
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)

    # Get confusion matrix
    cm = metrics.confusion_matrix
    cm_norm = metrics.confusion_matrix_normalized

    print(f"\nConfusion Matrix Shape: {cm.shape}")
    total_samples = sum(metrics.per_class_support.values())
    print(f"Total test samples: {total_samples}")

    # Calculate misclassifications from confusion matrix
    correct = np.trace(cm)
    misclassified = total_samples - correct
    misclassification_rate = misclassified / total_samples
    print(f"\nCorrect predictions: {correct}/{total_samples}")
    print(
        f"Misclassified samples: {misclassified}/{total_samples} ({misclassification_rate:.2%})"
    )

    # Find most confused classes
    print("\nMost confused class pairs (normalized confusion > 0.1):")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            if i != j and cm_norm[i, j] > 0.1:
                print(
                    f"  {class_names[i]} → {class_names[j]}: {cm_norm[i, j]:.2%} ({cm[i, j]} samples)"
                )

    # Calculate per-class accuracy from confusion matrix
    print("\n" + "=" * 60)
    print("PER-CLASS PERFORMANCE ANALYSIS")
    print("=" * 60)

    for i, class_name in enumerate(class_names):
        total_class_samples = cm[i].sum()
        correct_class = cm[i, i]
        class_accuracy = (
            correct_class / total_class_samples if total_class_samples > 0 else 0
        )
        print(
            f"{class_name}: {correct_class}/{total_class_samples} correct ({class_accuracy:.2%})"
        )

    print("\n" + "=" * 60)
    print("Evaluation complete! Metrics saved to 'evaluation_results/' directory.")
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
