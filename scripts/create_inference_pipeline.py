#!/usr/bin/env python3
"""
Create a complete inference pipeline with all feature engineering components.
This script properly saves the TF-IDF vectorizer and other encoders.
"""

import sys
import os
from pathlib import Path
import joblib
import json
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import LogDataLoader
from src.data.features import LogFeatureEngineer
from src.models.classifier import LogClassifier
from src.inference.predictor import LogPredictor


def create_inference_pipeline(
    model_path: str,
    dataset_path: str,
    output_path: str = "models/inference_pipeline.joblib",
) -> None:
    """
    Create a complete inference pipeline with properly saved feature engineering components.

    Args:
        model_path: Path to trained model
        dataset_path: Path to dataset CSV
        output_path: Path to save inference pipeline
    """
    print("Creating inference pipeline...")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")

    # 1. Load the trained model
    print("\n1. Loading trained model...")
    classifier = LogClassifier.load(model_path)
    print(f"   Model type: {classifier.config.model_type}")
    print(
        f"   Classes: {len(classifier.classes_) if classifier.classes_ is not None else 0}"
    )

    # 2. Load dataset and create feature engineer with same parameters as training
    print("\n2. Creating feature engineer from dataset...")
    loader = LogDataLoader(dataset_path)
    df = loader.load_data()

    # Create feature engineer
    engineer = LogFeatureEngineer(df)

    # Create features with same parameters as training
    # Based on train_model.py, these are the parameters used:
    # tfidf_max_features=100, tfidf_min_df=1, tfidf_max_df=1.0
    features, feature_names = engineer.create_all_features(
        tfidf_max_features=100,
        tfidf_min_df=1,
        tfidf_max_df=1.0,
    )

    print(f"   Created features with shape: {features.shape}")
    print(f"   Feature names: {len(feature_names)}")
    print(f"   TF-IDF vectorizer: {engineer.tfidf_vectorizer is not None}")
    print(f"   Label encoder: {engineer.label_encoder is not None}")
    print(f"   Service encoder: {engineer.service_encoder is not None}")

    # 3. Create predictor with model and feature engineer
    print("\n3. Creating predictor...")

    # Create a predictor instance
    predictor = LogPredictor.__new__(LogPredictor)
    predictor.model = classifier.model
    predictor.feature_engineer = engineer
    predictor.root_cause_labels = (
        [f"RC-{i + 1:02d}" for i in range(len(classifier.classes_))]
        if classifier.classes_ is not None
        else [f"RC-{i:02d}" for i in range(1, 9)]
    )
    predictor.model_path = Path(model_path)
    predictor.feature_engineer_path = None
    predictor.dataset_path = Path(dataset_path)

    # Import SummaryGenerator
    from src.inference.summary import SummaryGenerator

    predictor.summary_generator = SummaryGenerator()

    print(
        f"   Predictor created with {len(predictor.root_cause_labels)} root cause labels"
    )

    # 4. Save the complete pipeline
    print("\n4. Saving inference pipeline...")
    predictor.save(output_path)

    # 5. Verify the saved pipeline
    print("\n5. Verifying saved pipeline...")
    try:
        loaded_data = joblib.load(output_path)
        print(f"   Keys in saved file: {list(loaded_data.keys())}")

        if "feature_engineer" in loaded_data:
            fe = loaded_data["feature_engineer"]
            print(f"   Feature engineer type: {type(fe)}")
            print(f"   TF-IDF vectorizer exists: {hasattr(fe, 'tfidf_vectorizer')}")
            if hasattr(fe, "tfidf_vectorizer"):
                print(f"   TF-IDF vectorizer is None: {fe.tfidf_vectorizer is None}")
                if fe.tfidf_vectorizer is not None:
                    print(
                        f"   TF-IDF vocabulary size: {len(fe.tfidf_vectorizer.vocabulary_)}"
                    )

        print(f"\n✓ Inference pipeline saved successfully to {output_path}")

    except Exception as e:
        print(f"   Error verifying saved pipeline: {e}")
        raise


def main():
    """Main function to create inference pipeline."""
    # Paths
    model_path = "models/log_classifier_random_forest.joblib"
    dataset_path = "docs/log_dataset.csv"
    output_path = "models/inference_pipeline.joblib"

    # Check if files exist
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using scripts/train_model.py")
        return

    if not Path(dataset_path).exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Create inference pipeline
    try:
        create_inference_pipeline(model_path, dataset_path, output_path)
    except Exception as e:
        print(f"Failed to create inference pipeline: {e}")
        import traceback

        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
