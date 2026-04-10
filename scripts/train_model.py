#!/usr/bin/env python3
"""
Training script for multi-class classification model.
Integrates with feature engineering pipeline.
"""

import sys
import os
from pathlib import Path
import json
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import LogDataLoader
from src.data.features import LogFeatureEngineer
from src.models.classifier import LogClassifier, ModelConfig


def train_model(
    data_path: str,
    model_type: str = "random_forest",
    output_dir: str = "models",
    test_size: float = 0.3,
    random_state: int = 42,
) -> None:
    """
    Train classification model with full pipeline.

    Args:
        data_path: Path to CSV dataset
        model_type: Type of model to train ("logistic_regression", "random_forest", "xgboost")
        output_dir: Directory to save trained model and results
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Training {model_type} model for root cause classification...")
    print(f"Data: {data_path}")
    print(f"Output: {output_path}")

    # 1. Load and clean data
    print("\n1. Loading and cleaning data...")
    loader = LogDataLoader(data_path)
    df = loader.load_data()

    # 2. Feature engineering
    print("2. Feature engineering...")
    engineer = LogFeatureEngineer(df)

    # Create features
    features, feature_names = engineer.create_all_features(
        tfidf_max_features=100,  # Appropriate for small dataset
        tfidf_min_df=1,
        tfidf_max_df=1.0,  # Important for small dataset
    )

    # Prepare labels
    labels = engineer.prepare_labels()

    # Split data
    X_train, X_test, y_train, y_test = engineer.split_data(
        features, labels, test_size=test_size, random_state=random_state
    )

    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Feature names: {len(feature_names)}")

    # Save feature names
    with open(output_path / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # 3. Train model
    print(f"\n3. Training {model_type} model...")

    config = ModelConfig(
        model_type=model_type,
        random_state=random_state,
        class_weight="balanced",
        tune_hyperparameters=True,
        cv_folds=3,  # Smaller CV for small dataset
    )

    classifier = LogClassifier(config=config)
    classifier.fit(X_train, y_train, feature_names=feature_names)

    # 4. Evaluate model
    print("4. Evaluating model...")
    metrics = classifier.evaluate(X_test, y_test, return_report=True)

    # Save evaluation results
    with open(output_path / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 5. Save model
    print("5. Saving model...")
    model_path = output_path / f"log_classifier_{model_type}.joblib"
    classifier.save(model_path)

    # 6. Generate summary
    print("6. Generating summary...")
    config_summary = classifier.get_config_summary()
    feature_importance = classifier.get_feature_importance()

    summary = {
        "model_type": model_type,
        "model_path": str(model_path),
        "dataset": {
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "features": X_train.shape[1],
            "classes": len(classifier.classes_)
            if classifier.classes_ is not None
            else 0,
        },
        "config": config_summary,
        "evaluation": {
            "accuracy": metrics["accuracy"],
            "per_class_metrics": metrics.get("classification_report", {}),
        },
        "feature_importance": feature_importance,
    }

    with open(output_path / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 7. Print results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {model_path}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")

    if "classification_report" in metrics:
        report = metrics["classification_report"]
        print("\nPer-class F1 scores:")
        for class_name, scores in report.items():
            if class_name not in ["accuracy", "macro avg", "weighted avg", "micro avg"]:
                f1 = scores.get("f1-score", 0)
                support = scores.get("support", 0)
                print(f"  {class_name}: F1={f1:.3f} (support={support})")

    print(f"\nMacro F1: {report.get('macro avg', {}).get('f1-score', 0):.3f}")
    print(f"Weighted F1: {report.get('weighted avg', {}).get('f1-score', 0):.3f}")

    if feature_importance:
        print("\nTop 10 features by importance:")
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.4f}")

    print("\nSummary saved to:", output_path / "training_summary.json")


def compare_models(
    data_path: str,
    output_dir: str = "models",
    test_size: float = 0.3,
    random_state: int = 42,
) -> None:
    """
    Compare multiple model types and save results.

    Args:
        data_path: Path to CSV dataset
        output_dir: Directory to save results
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    """
    model_types = ["logistic_regression", "random_forest"]

    # Try XGBoost if available
    try:
        import xgboost

        model_types.append("xgboost")
    except ImportError:
        print("XGBoost not available. Skipping XGBoost comparison.")

    results = {}

    for model_type in model_types:
        print(f"\n{'=' * 60}")
        print(f"Training {model_type}")
        print("=" * 60)

        try:
            # Create model-specific output directory
            model_output_dir = Path(output_dir) / model_type
            model_output_dir.mkdir(exist_ok=True, parents=True)

            # Load data and engineer features once
            if "loader" not in locals():
                loader = LogDataLoader(data_path)
                df = loader.load_data()
                engineer = LogFeatureEngineer(df)

                # Create features
                features, feature_names = engineer.create_all_features(
                    tfidf_max_features=100, tfidf_min_df=1, tfidf_max_df=1.0
                )

                # Prepare labels
                labels = engineer.prepare_labels()

                # Split data
                X_train, X_test, y_train, y_test = engineer.split_data(
                    features, labels, test_size=test_size, random_state=random_state
                )

            # Train model
            config = ModelConfig(
                model_type=model_type,
                random_state=random_state,
                class_weight="balanced",
                tune_hyperparameters=True,
                cv_folds=3,
            )

            classifier = LogClassifier(config=config)
            classifier.fit(X_train, y_train, feature_names=feature_names)

            # Evaluate
            metrics = classifier.evaluate(X_test, y_test, return_report=True)

            # Save model
            model_path = model_output_dir / f"log_classifier_{model_type}.joblib"
            classifier.save(model_path)

            # Store results
            results[model_type] = {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["classification_report"]["macro avg"]["f1-score"],
                "weighted_f1": metrics["classification_report"]["weighted avg"][
                    "f1-score"
                ],
                "model_path": str(model_path),
            }

            # Save individual results
            with open(model_output_dir / "results.json", "w") as f:
                json.dump(
                    {
                        "model_type": model_type,
                        "accuracy": metrics["accuracy"],
                        "classification_report": metrics["classification_report"],
                        "config": classifier.get_config_summary(),
                    },
                    f,
                    indent=2,
                )

            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(
                f"  Macro F1: {metrics['classification_report']['macro avg']['f1-score']:.4f}"
            )

        except Exception as e:
            print(f"  Error training {model_type}: {e}")
            results[model_type] = {"error": str(e)}

    # Save comparison results
    comparison_path = Path(output_dir) / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)

    for model_type, result in results.items():
        if "error" not in result:
            print(f"{model_type}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Macro F1: {result['macro_f1']:.4f}")
            print(f"  Weighted F1: {result['weighted_f1']:.4f}")
        else:
            print(f"{model_type}: ERROR - {result['error']}")

    print(f"\nComparison saved to: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train multi-class classification model for root cause prediction"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="docs/Flutterwave AI Engineer Assessment Dataset.xlsx - log_dataset.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logistic_regression", "random_forest", "xgboost"],
        default="random_forest",
        help="Type of model to train",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model and results",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.3, help="Proportion of data for testing"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare multiple model types"
    )

    args = parser.parse_args()

    if args.compare:
        compare_models(
            data_path=args.data,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    else:
        train_model(
            data_path=args.data,
            model_type=args.model_type,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()
