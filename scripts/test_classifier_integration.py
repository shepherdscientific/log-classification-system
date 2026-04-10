#!/usr/bin/env python3
"""
Test classifier integration with feature engineering.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import LogDataLoader
from src.data.features import LogFeatureEngineer
from src.models.classifier import LogClassifier, ModelConfig
import numpy as np


def test_integration():
    """Test the full integration pipeline."""
    print("Testing classifier integration pipeline...")

    # 1. Load data
    print("\n1. Loading data...")
    data_path = "docs/Flutterwave AI Engineer Assessment Dataset.xlsx - log_dataset.csv"
    loader = LogDataLoader(data_path)
    df = loader.load_data()

    print(f"   Loaded {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")

    # 2. Feature engineering
    print("\n2. Feature engineering...")
    engineer = LogFeatureEngineer(df)

    # Create features
    features, feature_names = engineer.create_all_features(
        tfidf_max_features=50,  # Small for testing
        tfidf_min_df=1,
        tfidf_max_df=1.0,  # Important for small dataset
    )

    # Prepare labels
    labels = engineer.prepare_labels()

    print(f"   Features shape: {features.shape}")
    print(f"   Feature names: {len(feature_names)}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Unique classes: {np.unique(labels)}")

    # 3. Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = engineer.split_data(
        features, labels, test_size=0.3, random_state=42
    )

    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")

    # 4. Train classifier
    print("\n4. Training classifier...")
    config = ModelConfig(
        model_type="random_forest",
        random_state=42,
        class_weight="balanced",
        tune_hyperparameters=False,  # Disable for quick test
        cv_folds=3,
    )

    classifier = LogClassifier(config=config)
    classifier.fit(X_train, y_train, feature_names=feature_names)

    print("   Model trained successfully")

    # 5. Evaluate
    print("\n5. Evaluating model...")
    metrics = classifier.evaluate(X_test, y_test, return_report=True)

    print(f"   Accuracy: {metrics['accuracy']:.4f}")

    if "classification_report" in metrics:
        report = metrics["classification_report"]
        print(f"   Macro F1: {report.get('macro avg', {}).get('f1-score', 0):.3f}")

    # 6. Test predictions
    print("\n6. Testing predictions...")
    sample_idx = 0
    sample = X_test[sample_idx : sample_idx + 1]
    prediction = classifier.predict(sample)
    probability = classifier.predict_proba(sample)

    print(f"   Sample prediction: {prediction[0]}")
    print(f"   Probabilities shape: {probability.shape}")

    # 7. Test feature importance
    print("\n7. Testing feature importance...")
    importance = classifier.get_feature_importance()
    if importance:
        print(f"   Got feature importance for {len(importance)} features")
        # Show top 5
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]
        print("   Top 5 features:")
        for feature, imp in sorted_features:
            print(f"     {feature}: {imp:.4f}")

    # 8. Test serialization
    print("\n8. Testing serialization...")
    import tempfile
    import joblib

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        classifier.save(tmp_path)
        print(f"   Model saved to: {tmp_path}")

        loaded_classifier = LogClassifier.load(tmp_path)
        print("   Model loaded successfully")

        # Verify predictions match
        original_pred = classifier.predict(sample)
        loaded_pred = loaded_classifier.predict(sample)

        if np.array_equal(original_pred, loaded_pred):
            print("   Predictions match ✓")
        else:
            print("   Predictions don't match ✗")

    finally:
        import os

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_integration()
