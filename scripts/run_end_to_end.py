#!/usr/bin/env python3
"""
End-to-end integration pipeline for AI Log Classification System.

This script demonstrates the complete workflow:
1. Load and preprocess data
2. Train classification model
3. Evaluate model performance
4. Run inference on sample logs
5. Generate structured summaries
6. Save comprehensive evaluation results
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import LogDataLoader
from src.data.features import LogFeatureEngineer
from src.models.classifier import LogClassifier
from src.evaluation.metrics import MultiClassEvaluator
from src.inference.predictor import LogPredictor
from src.inference.summary import SummaryGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EndToEndPipeline:
    """Complete end-to-end pipeline for log classification system."""

    def __init__(self, data_path: str, output_dir: str = "evaluation_results"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.data_loader = None
        self.feature_engineer = None
        self.classifier = None
        self.evaluator = None
        self.predictor = None
        self.summary_generator = None

        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "data_info": {},
            "training_results": {},
            "evaluation_metrics": {},
            "inference_examples": [],
            "summary_examples": [],
        }

    def load_and_analyze_data(self):
        """Load dataset and perform initial analysis."""
        logger.info("Step 1: Loading and analyzing dataset...")

        self.data_loader = LogDataLoader(self.data_path)
        df = self.data_loader.load_data()

        # Store data info
        self.results["data_info"] = {
            "total_samples": len(df),
            "columns": list(df.columns),
            "root_cause_distribution": df["root_cause_label"].value_counts().to_dict(),
            "service_distribution": df["service"].value_counts().to_dict(),
            "severity_distribution": df["severity"].value_counts().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
        }

        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        logger.info(
            f"Root cause distribution: {self.results['data_info']['root_cause_distribution']}"
        )

        return df

    def preprocess_features(self, df):
        """Preprocess data and extract features."""
        logger.info("Step 2: Preprocessing features...")

        self.feature_engineer = LogFeatureEngineer(df)

        # Create all features
        features, feature_names = self.feature_engineer.create_all_features(
            tfidf_max_features=100,
            tfidf_max_df=1.0,  # Important for small datasets
        )

        # Prepare labels
        labels = self.feature_engineer.prepare_labels()

        # Split data
        X_train, X_test, y_train, y_test = self.feature_engineer.split_data(
            features, labels, test_size=0.3, random_state=42
        )

        logger.info(
            f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features"
        )
        logger.info(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """Train classification model."""
        logger.info("Step 3: Training classification model...")

        from src.models.classifier import ModelConfig

        # Create model configuration
        config = ModelConfig(
            model_type="random_forest",
            random_state=42,
            class_weight="balanced",
            cv_folds=3,  # Use fewer folds for small dataset
        )

        self.classifier = LogClassifier(config=config)

        # Train the model
        self.classifier.fit(X_train, y_train)

        # Store training results
        self.results["training_results"] = {
            "model_type": "random_forest",
            "best_params": self.classifier.model.get_params(),
            "config_summary": self.classifier.get_config_summary(),
        }

        logger.info(f"Model trained successfully")
        logger.info(f"Model type: {config.model_type}")

        return self.classifier

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance."""
        logger.info("Step 4: Evaluating model performance...")

        y_pred = self.classifier.predict(X_test)

        # Get class names from label encoder
        class_names = list(self.feature_engineer.label_encoder.classes_)

        self.evaluator = MultiClassEvaluator(class_names=class_names)

        # Calculate all metrics
        metrics = self.evaluator.compute_metrics(y_test, y_pred)

        # Store evaluation metrics
        self.results["evaluation_metrics"] = {
            "accuracy": metrics.accuracy,
            "macro_precision": metrics.macro_precision,
            "macro_recall": metrics.macro_recall,
            "macro_f1": metrics.macro_f1,
            "per_class_precision": metrics.per_class_precision,
            "per_class_recall": metrics.per_class_recall,
            "per_class_f1": metrics.per_class_f1,
            "confusion_matrix": metrics.confusion_matrix.tolist(),
        }

        # Generate visualizations
        confusion_matrix_path = self.output_dir / "confusion_matrix.png"
        self.evaluator.plot_confusion_matrix(metrics, str(confusion_matrix_path))

        # Save classification report
        report_path = self.output_dir / "classification_report.txt"
        report_text = self.evaluator.generate_classification_report(metrics)
        with open(report_path, "w") as f:
            f.write(report_text)

        logger.info(f"Test accuracy: {metrics.accuracy:.3f}")
        logger.info(f"Macro F1-score: {metrics.macro_f1:.3f}")
        logger.info(f"Per-class F1 scores: {metrics.per_class_f1}")

        return metrics

    def save_model(self):
        """Save trained model and preprocessing pipeline."""
        logger.info("Step 5: Saving model artifacts...")

        model_dir = self.output_dir / "model_artifacts"
        model_dir.mkdir(exist_ok=True)

        # Save model
        model_path = model_dir / "log_classifier.pkl"
        self.classifier.save(str(model_path))

        # Save feature engineer components using joblib
        import joblib

        # Save TF-IDF vectorizer
        tfidf_path = model_dir / "tfidf_vectorizer.pkl"
        joblib.dump(self.feature_engineer.tfidf_vectorizer, str(tfidf_path))

        # Save label encoders
        label_encoder_path = model_dir / "label_encoder.pkl"
        joblib.dump(self.feature_engineer.label_encoder, str(label_encoder_path))

        service_encoder_path = model_dir / "service_encoder.pkl"
        joblib.dump(self.feature_engineer.service_encoder, str(service_encoder_path))

        severity_encoder_path = model_dir / "severity_encoder.pkl"
        joblib.dump(self.feature_engineer.severity_encoder, str(severity_encoder_path))

        # Also save a bundled predictor for easier loading
        from src.inference.predictor import LogPredictor

        # Create a predictor with the trained model
        bundled_predictor_path = model_dir / "log_predictor.pkl"

        # Create predictor data bundle
        predictor_data = {
            "model": self.classifier.model,
            "feature_engineer": self.feature_engineer,
            "root_cause_labels": [f"RC-{i:02d}" for i in range(1, 9)],
            "model_path": str(model_path),
        }

        joblib.dump(predictor_data, str(bundled_predictor_path))

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Feature engineering components saved to {model_dir}")
        logger.info(f"Bundled predictor saved to {bundled_predictor_path}")

        return str(model_path), str(model_dir)

    def run_inference_examples(self, df):
        """Run inference on sample logs from each root cause category."""
        logger.info("Step 6: Running inference examples...")

        # Get model path from saved model
        model_path = self.output_dir / "model_artifacts" / "log_classifier.pkl"

        # Initialize predictor with dataset path for feature engineering
        self.predictor = LogPredictor(
            model_path=str(model_path), dataset_path=self.data_path
        )

        # Load sample logs from each category
        sample_logs = []
        for rc in df["root_cause_label"].unique():
            sample = df[df["root_cause_label"] == rc].iloc[0]
            sample_logs.append(
                {
                    "log_id": sample["log_id"],
                    "timestamp": sample["timestamp"],
                    "service": sample["service"],
                    "severity": sample["severity"],
                    "log_message": sample["log_message"],
                    "true_label": sample["root_cause_label"],
                }
            )

        # Run inference
        predictions = []
        for log in sample_logs:
            prediction = self.predictor.predict_single(
                log_message=log["log_message"],
                service=log["service"],
                severity=log["severity"],
                timestamp=log["timestamp"],
            )

            result = {
                "log_id": log["log_id"],
                "true_label": log["true_label"],
                "predicted_label": prediction.root_cause,
                "confidence": prediction.confidence,
                "all_probabilities": prediction.top_n_predictions,
                "correct": log["true_label"] == prediction.root_cause,
            }

            predictions.append(result)
            self.results["inference_examples"].append(result)

            logger.info(
                f"Log {log['log_id']}: True={log['true_label']}, Predicted={prediction.root_cause}, Confidence={prediction.confidence:.3f}, Correct={result['correct']}"
            )

        # Calculate inference accuracy
        correct_predictions = sum(1 for p in predictions if p["correct"])
        inference_accuracy = (
            correct_predictions / len(predictions) if predictions else 0
        )

        logger.info(
            f"Inference accuracy on samples: {inference_accuracy:.3f} ({correct_predictions}/{len(predictions)})"
        )

        return predictions

    def generate_summaries(self, df):
        """Generate structured summaries for predictions."""
        logger.info("Step 7: Generating structured summaries...")

        self.summary_generator = SummaryGenerator()

        # Get sample logs for summary generation
        sample_logs = []
        for rc in df["root_cause_label"].unique():
            sample = df[df["root_cause_label"] == rc].iloc[0]
            sample_logs.append(
                {
                    "log_id": sample["log_id"],
                    "log_message": sample["log_message"],
                    "service": sample["service"],
                    "severity": sample["severity"],
                    "timestamp": sample["timestamp"],
                }
            )

        # Generate summaries
        summaries = []
        for log in sample_logs:
            # First get prediction
            prediction = self.predictor.predict_single(
                log_message=log["log_message"],
                service=log["service"],
                severity=log["severity"],
                timestamp=log["timestamp"],
            )

            # Generate summary using the prediction result
            summary = prediction.summary

            if summary:
                summaries.append(summary)
                self.results["summary_examples"].append(summary.to_dict())

                logger.info(f"Summary for {log['log_id']}: {summary.root_cause}")
                logger.info(f"  Confidence: {summary.confidence:.3f}")
                logger.info(f"  Severity: {summary.severity}")
                logger.info(f"  Summary: {summary.summary[:100]}...")
            else:
                logger.warning(f"No summary generated for {log['log_id']}")

        return summaries

    def save_results(self):
        """Save all results to JSON file."""
        logger.info("Step 8: Saving comprehensive results...")

        results_path = self.output_dir / "end_to_end_results.json"

        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_path}")

        return str(results_path)

    def run(self):
        """Execute complete end-to-end pipeline."""
        logger.info("=" * 60)
        logger.info("Starting End-to-End Pipeline")
        logger.info("=" * 60)

        try:
            # Step 1: Load data
            df = self.load_and_analyze_data()

            # Step 2: Preprocess features
            X_train, X_test, y_train, y_test = self.preprocess_features(df)

            # Step 3: Train model
            self.train_model(X_train, y_train)

            # Step 4: Evaluate model
            self.evaluate_model(X_test, y_test)

            # Step 5: Save model
            self.save_model()

            # Step 6: Run inference examples
            self.run_inference_examples(df)

            # Step 7: Generate summaries
            self.generate_summaries(df)

            # Step 8: Save results
            results_path = self.save_results()

            logger.info("=" * 60)
            logger.info("Pipeline completed successfully!")
            logger.info(f"Results saved to: {results_path}")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False


def main():
    """Main entry point."""
    # Path to dataset
    data_path = "docs/Flutterwave AI Engineer Assessment Dataset.xlsx - log_dataset.csv"

    # Check if dataset exists
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at: {data_path}")
        logger.info("Please ensure the dataset file exists in the docs/ directory")
        return 1

    # Create and run pipeline
    pipeline = EndToEndPipeline(data_path)
    success = pipeline.run()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
