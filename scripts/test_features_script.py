#!/usr/bin/env python3
"""Test script for feature engineering pipeline."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from src.data.loader import LogDataLoader
from src.data.features import LogFeatureEngineer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_feature_engineering():
    """Test the complete feature engineering pipeline."""
    logger.info("Starting feature engineering test")

    # Load data
    data_path = "docs/log_dataset.csv"
    loader = LogDataLoader(data_path)
    df = loader.load_data()

    # Validate data
    validation = loader.validate_data()
    logger.info(f"Data validation: {validation}")

    # Create feature engineer
    feature_engineer = LogFeatureEngineer(df)

    # Test text preprocessing
    logger.info("Testing text preprocessing...")
    df_with_text_features = feature_engineer.create_text_features()
    logger.info(
        f"Created text features. Sample clean message: {df_with_text_features['clean_message'].iloc[0][:100]}..."
    )
    logger.info(
        f"Error type distribution: {df_with_text_features['error_type'].value_counts().to_dict()}"
    )

    # Test TF-IDF features
    logger.info("Testing TF-IDF features...")
    tfidf_features = feature_engineer.create_tfidf_features(max_features=500)
    logger.info(f"TF-IDF features shape: {tfidf_features.shape}")

    # Test categorical features
    logger.info("Testing categorical features...")
    categorical_features = feature_engineer.create_categorical_features()
    logger.info(f"Categorical features shape: {categorical_features.shape}")

    # Test timestamp features
    logger.info("Testing timestamp features...")
    timestamp_features = feature_engineer.extract_timestamp_features()
    logger.info(f"Timestamp features shape: {timestamp_features.shape}")

    # Test all features combined
    logger.info("Testing all features combined...")
    all_features, feature_names = feature_engineer.create_all_features(
        tfidf_max_features=500
    )
    logger.info(f"All features shape: {all_features.shape}")
    logger.info(f"Number of feature names: {len(feature_names)}")
    logger.info(f"First 10 feature names: {feature_names[:10]}")

    # Test label preparation
    logger.info("Testing label preparation...")
    labels = feature_engineer.prepare_labels()
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Label classes: {feature_engineer.label_encoder.classes_}")

    # Test data splitting
    logger.info("Testing data splitting...")
    X_train, X_test, y_train, y_test = feature_engineer.split_data(
        all_features, labels, test_size=0.2, random_state=42
    )
    logger.info(f"Train features shape: {X_train.shape}")
    logger.info(f"Test features shape: {X_test.shape}")
    logger.info(f"Train labels shape: {y_train.shape}")
    logger.info(f"Test labels shape: {y_test.shape}")

    # Test feature analysis
    logger.info("Testing feature analysis...")
    analysis = feature_engineer.get_feature_analysis()
    logger.info(f"Feature analysis: {analysis}")

    # Verify class distribution in splits
    train_classes = np.unique(y_train, return_counts=True)
    test_classes = np.unique(y_test, return_counts=True)

    logger.info(
        f"Train class distribution: {dict(zip(train_classes[0], train_classes[1]))}"
    )
    logger.info(
        f"Test class distribution: {dict(zip(test_classes[0], test_classes[1]))}"
    )

    # Check that we have all 8 classes in both splits
    assert len(np.unique(y_train)) == 8, (
        f"Expected 8 classes in train, got {len(np.unique(y_train))}"
    )
    assert len(np.unique(y_test)) == 8, (
        f"Expected 8 classes in test, got {len(np.unique(y_test))}"
    )

    logger.info("All tests passed!")

    return {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
        "feature_names_count": len(feature_names),
        "analysis": analysis,
    }


if __name__ == "__main__":
    try:
        results = test_feature_engineering()
        print("\n" + "=" * 50)
        print("FEATURE ENGINEERING TEST RESULTS")
        print("=" * 50)
        print(f"Train features: {results['X_train_shape']}")
        print(f"Test features: {results['X_test_shape']}")
        print(f"Train labels: {results['y_train_shape']}")
        print(f"Test labels: {results['y_test_shape']}")
        print(f"Total features: {results['feature_names_count']}")
        print("\nFeature Analysis Summary:")
        for category, details in results["analysis"].items():
            print(f"  {category}: {details}")
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
