#!/usr/bin/env python3
"""
Root cause analysis demonstration script.

This script demonstrates the root cause analysis capabilities:
1. Load trained model and data
2. Perform misclassification analysis
3. Analyze feature importance for each root cause
4. Generate insights and recommendations
5. Save comprehensive analysis results
"""

import os
import sys
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import LogDataLoader
from src.data.features import LogFeatureEngineer
from src.evaluation.analysis import RootCauseAnalyzer
from src.models.classifier import LogClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RootCauseAnalysisPipeline:
    """Pipeline for root cause analysis and insights generation."""

    def __init__(
        self, data_path: str, model_path: str, output_dir: str = "analysis_results"
    ):
        self.data_path = data_path
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.data_loader: Optional[LogDataLoader] = None
        self.feature_engineer: Optional[LogFeatureEngineer] = None
        self.classifier: Optional[LogClassifier] = None
        self.analyzer: Optional[RootCauseAnalyzer] = None

        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "data_info": {},
            "analysis_results": {},
        }

    def load_data(self):
        """Load and preprocess data."""
        logger.info("Loading data from %s", self.data_path)

        self.data_loader = LogDataLoader(self.data_path)
        df = self.data_loader.load()

        # Store data info
        self.results["data_info"] = {
            "total_samples": len(df),
            "columns": list(df.columns),
            "root_cause_distribution": dict(df["root_cause_label"].value_counts()),
            "service_distribution": dict(df["service"].value_counts()),
            "severity_distribution": dict(df["severity"].value_counts()),
        }

        logger.info(
            "Loaded %d samples with %d root cause categories",
            len(df),
            len(df["root_cause_label"].unique()),
        )

        return df

    def prepare_features(self, df: pd.DataFrame):
        """Prepare features for analysis."""
        logger.info("Preparing features")

        self.feature_engineer = LogFeatureEngineer(df)
        X, feature_names = self.feature_engineer.create_all_features()
        y = self.feature_engineer.prepare_labels()

        # Split data (using same split as training for consistency)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        logger.info("Feature engineering complete: %d features", len(feature_names))
        logger.info(
            "Train set: %d samples, Test set: %d samples", len(X_train), len(X_test)
        )

        return X_train, X_test, y_train, y_test, feature_names

    def load_model(self):
        """Load trained model."""
        logger.info("Loading model from %s", self.model_path)

        self.classifier = LogClassifier()

        if self.model_path.suffix == ".pkl":
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
                self.classifier.model = model_data["model"]
                self.classifier.class_names = model_data["class_names"]
                self.classifier.feature_names = model_data.get("feature_names", [])
        else:
            # Assume it's a directory with model artifacts
            self.classifier.load(self.model_path)

        logger.info("Model loaded: %s", type(self.classifier.model).__name__)

        return self.classifier

    def perform_analysis(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        feature_names: list,
        sample_data: pd.DataFrame,
    ):
        """Perform comprehensive root cause analysis."""
        logger.info("Performing root cause analysis")

        # Initialize analyzer with class names
        class_names = [f"RC-{i + 1:02d}" for i in range(len(np.unique(y_test)))]
        self.analyzer = RootCauseAnalyzer(class_names=class_names)

        # Generate comprehensive insights
        if self.classifier is None or self.classifier.model is None:
            raise ValueError("Classifier not properly loaded")
        insights = self.analyzer.generate_insights(
            y_true=y_test,
            y_pred=y_pred,
            model=self.classifier.model,
            X=X_test,
            sample_data=sample_data,
            feature_names=feature_names,
        )

        # Store analysis results
        self.results["analysis_results"] = insights.to_dict()

        # Add additional analysis metrics
        self.results["analysis_summary"] = {
            "overall_accuracy": insights.overall_accuracy,
            "most_challenging_class": insights.most_challenging_class,
            "easiest_class": insights.easiest_class,
            "total_recommendations": len(insights.recommendations),
            "total_patterns_identified": len(insights.identified_patterns),
        }

        logger.info("Analysis complete:")
        logger.info("  Overall accuracy: %.2f%%", insights.overall_accuracy * 100)
        logger.info("  Most challenging class: %s", insights.most_challenging_class)
        logger.info("  Easiest class: %s", insights.easiest_class)
        logger.info("  Generated %d recommendations", len(insights.recommendations))
        logger.info("  Identified %d patterns", len(insights.identified_patterns))

        return insights

    def save_results(self):
        """Save analysis results to files."""
        logger.info("Saving analysis results to %s", self.output_dir)

        # Save comprehensive results
        results_path = self.output_dir / "root_cause_analysis_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save recommendations separately
        recommendations = self.results["analysis_results"].get("recommendations", [])
        if recommendations:
            recs_path = self.output_dir / "recommendations.json"
            with open(recs_path, "w") as f:
                json.dump(recommendations, f, indent=2)

        # Save patterns separately
        patterns = self.results["analysis_results"].get("identified_patterns", [])
        if patterns:
            patterns_path = self.output_dir / "identified_patterns.json"
            with open(patterns_path, "w") as f:
                json.dump(patterns, f, indent=2)

        # Generate summary report
        self._generate_summary_report()

        logger.info("Results saved to %s", self.output_dir)

    def _generate_summary_report(self):
        """Generate human-readable summary report."""
        report_path = self.output_dir / "analysis_summary.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ROOT CAUSE ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Basic info
            f.write(f"Analysis Date: {self.results['timestamp']}\n")
            f.write(f"Total Samples: {self.results['data_info']['total_samples']}\n")
            f.write(
                f"Root Cause Categories: {len(self.results['data_info']['root_cause_distribution'])}\n\n"
            )

            # Performance summary
            summary = self.results.get("analysis_summary", {})
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall Accuracy: {summary.get('overall_accuracy', 0):.1%}\n")
            f.write(
                f"Most Challenging Class: {summary.get('most_challenging_class', 'N/A')}\n"
            )
            f.write(f"Easiest Class: {summary.get('easiest_class', 'N/A')}\n\n")

            # Top recommendations
            recommendations = self.results["analysis_results"].get(
                "recommendations", []
            )
            if recommendations:
                f.write("TOP RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                for i, rec in enumerate(recommendations[:5], 1):
                    f.write(
                        f"{i}. [{rec.get('priority', 'medium').upper()}] {rec.get('description', '')}\n"
                    )
                    f.write(f"   Action: {rec.get('action', '')}\n\n")

            # Key patterns
            patterns = self.results["analysis_results"].get("identified_patterns", [])
            if patterns:
                f.write("KEY PATTERNS IDENTIFIED\n")
                f.write("-" * 40 + "\n")
                for i, pattern in enumerate(patterns[:3], 1):
                    f.write(
                        f"{i}. {pattern.get('type', 'Unknown')}: {pattern.get('description', '')}\n"
                    )
                    f.write(f"   Implication: {pattern.get('implication', '')}\n\n")

            # Challenging pairs
            misanalysis = self.results["analysis_results"].get(
                "misclassification_analysis", {}
            )
            challenging_pairs = misanalysis.get("challenging_pairs", [])
            if challenging_pairs:
                f.write("MOST CHALLENGING CLASS PAIRS\n")
                f.write("-" * 40 + "\n")
                for i, pair in enumerate(challenging_pairs[:3], 1):
                    f.write(
                        f"{i}. {pair.get('pair', '')}: {pair.get('count', 0)} misclassifications "
                        f"({pair.get('error_rate', 0):.1%} error rate)\n"
                    )
                    f.write(f"   {pair.get('description', '')}\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

    def run(self):
        """Run the complete analysis pipeline."""
        logger.info("Starting root cause analysis pipeline")

        try:
            # Step 1: Load data
            df = self.load_data()

            # Step 2: Prepare features
            X_train, X_test, y_train, y_test, feature_names = self.prepare_features(df)

            # Get sample data for test set (for analysis)
            _, test_indices = train_test_split(
                df.index,
                test_size=0.3,
                random_state=42,
                stratify=df["root_cause_label"],
            )
            sample_data = df.iloc[test_indices].reset_index(drop=True)

            # Step 3: Load model
            self.load_model()

            # Step 4: Make predictions
            logger.info("Making predictions on test set")
            if self.classifier is None or self.classifier.model is None:
                raise ValueError("Classifier not properly loaded")
            y_pred = self.classifier.model.predict(X_test)
            y_proba = None
            if hasattr(self.classifier.model, "predict_proba"):
                y_proba = self.classifier.model.predict_proba(X_test)

            # Step 5: Perform analysis
            insights = self.perform_analysis(
                X_test, y_test, y_pred, feature_names, sample_data
            )

            # Step 6: Save results
            self.save_results()

            logger.info("Root cause analysis pipeline completed successfully")
            return insights

        except Exception as e:
            logger.error("Error in analysis pipeline: %s", str(e))
            raise


def main():
    """Main function to run the analysis pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Run root cause analysis pipeline")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to dataset CSV file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model file or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_results",
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = RootCauseAnalysisPipeline(
        data_path=args.data, model_path=args.model, output_dir=args.output
    )

    try:
        pipeline.run()
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {args.output}/")
        print(f"  - root_cause_analysis_results.json (comprehensive results)")
        print(f"  - analysis_summary.txt (human-readable summary)")
        print(f"  - recommendations.json (actionable recommendations)")
        print(f"  - identified_patterns.json (key patterns identified)")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
