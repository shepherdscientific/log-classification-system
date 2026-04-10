#!/usr/bin/env python3
"""
Script to demonstrate comprehensive multi-class evaluation metrics.
Loads trained model and test data, computes all metrics, and saves results.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import MultiClassEvaluator
from src.data.loader import LogDataLoader
from src.data.features import LogFeatureEngineer
from src.models.classifier import LogClassifier
import joblib


def load_test_data():
    """Load test data for evaluation."""
    print("Loading test data...")

    # Load dataset
    dataset_path = "docs/log_dataset.csv"
    data_loader = LogDataLoader(dataset_path)
    df = data_loader.load_data()

    # Create features
    feature_engineer = LogFeatureEngineer(df)
    X, feature_names = feature_engineer.create_all_features()
    y = feature_engineer.prepare_labels()

    # Split data (using same random state as training)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_test, y_test, feature_names


def load_trained_model():
    """Load trained model from file."""
    print("Loading trained model...")

    model_path = "models_compare/random_forest/log_classifier_random_forest.joblib"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model_data = joblib.load(model_path)
    return model_data


def run_evaluation():
    """Run comprehensive evaluation."""
    print("=" * 60)
    print("Multi-class Evaluation Metrics Demo")
    print("=" * 60)

    # Load test data
    X_test, y_test, feature_names = load_test_data()

    # Load trained model
    model_data = load_trained_model()
    model = model_data["model"]

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Class names for RC-01 to RC-08
    class_names = [f"RC-{i + 1:02d}" for i in range(8)]

    # Create evaluator
    evaluator = MultiClassEvaluator(class_names=class_names)

    # Compute comprehensive metrics
    print("\nComputing comprehensive metrics...")
    metrics = evaluator.compute_metrics(y_test, y_pred, y_proba)

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
        metrics, output_dir, prefix="log_classification"
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
    print(f"Total test samples: {len(y_test)}")

    # Analyze misclassifications
    misclassified = np.sum(y_test != y_pred)
    misclassification_rate = misclassified / len(y_test)
    print(
        f"\nMisclassified samples: {misclassified}/{len(y_test)} ({misclassification_rate:.2%})"
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
