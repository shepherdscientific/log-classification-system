"""
Unit tests for multi-class evaluation metrics module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from src.evaluation.metrics import MultiClassEvaluator, MultiClassMetrics


class TestMultiClassMetrics:
    """Test MultiClassMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating MultiClassMetrics object."""
        metrics = MultiClassMetrics(
            accuracy=0.85,
            macro_precision=0.82,
            macro_recall=0.83,
            macro_f1=0.825,
            weighted_precision=0.84,
            weighted_recall=0.85,
            weighted_f1=0.845,
            per_class_precision={"Class1": 0.9, "Class2": 0.8},
            per_class_recall={"Class1": 0.85, "Class2": 0.75},
            per_class_f1={"Class1": 0.875, "Class2": 0.775},
            per_class_support={"Class1": 10, "Class2": 15},
            confusion_matrix=np.array([[8, 2], [3, 12]]),
            confusion_matrix_normalized=np.array([[0.8, 0.2], [0.2, 0.8]]),
        )

        assert metrics.accuracy == 0.85
        assert metrics.macro_precision == 0.82
        assert metrics.per_class_precision["Class1"] == 0.9
        assert metrics.confusion_matrix.shape == (2, 2)

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = MultiClassMetrics(
            accuracy=0.85,
            macro_precision=0.82,
            macro_recall=0.83,
            macro_f1=0.825,
            weighted_precision=0.84,
            weighted_recall=0.85,
            weighted_f1=0.845,
            per_class_precision={"Class1": 0.9},
            per_class_recall={"Class1": 0.85},
            per_class_f1={"Class1": 0.875},
            per_class_support={"Class1": 10},
            confusion_matrix=np.array([[8, 2]]),
            confusion_matrix_normalized=np.array([[0.8, 0.2]]),
        )

        metrics_dict = metrics.to_dict()
        assert metrics_dict["accuracy"] == 0.85
        assert metrics_dict["macro_precision"] == 0.82
        assert "per_class_precision" in metrics_dict
        assert "confusion_matrix" in metrics_dict

    def test_to_dataframe(self):
        """Test converting metrics to pandas DataFrame."""
        metrics = MultiClassMetrics(
            accuracy=0.85,
            macro_precision=0.82,
            macro_recall=0.83,
            macro_f1=0.825,
            weighted_precision=0.84,
            weighted_recall=0.85,
            weighted_f1=0.845,
            per_class_precision={"Class1": 0.9, "Class2": 0.8},
            per_class_recall={"Class1": 0.85, "Class2": 0.75},
            per_class_f1={"Class1": 0.875, "Class2": 0.775},
            per_class_support={"Class1": 10, "Class2": 15},
            confusion_matrix=np.array([[8, 2], [3, 12]]),
            confusion_matrix_normalized=np.array([[0.8, 0.2], [0.2, 0.8]]),
        )

        df = metrics.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1" in df.columns
        assert "support" in df.columns

    def test_save_and_load(self):
        """Test saving metrics to file and loading back."""
        metrics = MultiClassMetrics(
            accuracy=0.85,
            macro_precision=0.82,
            macro_recall=0.83,
            macro_f1=0.825,
            weighted_precision=0.84,
            weighted_recall=0.85,
            weighted_f1=0.845,
            per_class_precision={"Class1": 0.9},
            per_class_recall={"Class1": 0.85},
            per_class_f1={"Class1": 0.875},
            per_class_support={"Class1": 10},
            confusion_matrix=np.array([[8, 2]]),
            confusion_matrix_normalized=np.array([[0.8, 0.2]]),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_metrics.json"
            metrics.save(filepath)

            # Load back and verify
            with open(filepath, "r") as f:
                loaded_data = json.load(f)

            assert loaded_data["accuracy"] == 0.85
            assert loaded_data["macro_precision"] == 0.82
            assert loaded_data["per_class_precision"]["Class1"] == 0.9


class TestMultiClassEvaluator:
    """Test MultiClassEvaluator class."""

    def setup_method(self):
        """Set up test data."""
        # Create synthetic test data for 3 classes
        np.random.seed(42)
        self.n_samples = 100
        self.n_classes = 3

        # True labels with some imbalance
        self.y_true = np.random.choice(
            [0, 1, 2], size=self.n_samples, p=[0.4, 0.3, 0.3]
        )

        # Predicted labels (simulating 80% accuracy)
        self.y_pred = self.y_true.copy()
        # Introduce some errors
        error_indices = np.random.choice(self.n_samples, size=20, replace=False)
        for idx in error_indices:
            # Predict a different class
            possible_classes = [
                c for c in range(self.n_classes) if c != self.y_true[idx]
            ]
            self.y_pred[idx] = np.random.choice(possible_classes)

        # Simulated probabilities
        self.y_proba = np.random.rand(self.n_samples, self.n_classes)
        # Normalize to make valid probabilities
        self.y_proba = self.y_proba / self.y_proba.sum(axis=1, keepdims=True)
        # Make sure highest probability aligns with predictions
        for i in range(self.n_samples):
            self.y_proba[i, self.y_pred[i]] = np.max(self.y_proba[i]) + 0.1
            self.y_proba[i] = self.y_proba[i] / self.y_proba[i].sum()

        self.class_names = ["Class0", "Class1", "Class2"]

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = MultiClassEvaluator(class_names=self.class_names)
        assert evaluator.class_names == self.class_names

        evaluator_no_names = MultiClassEvaluator()
        assert evaluator_no_names.class_names is None

    def test_compute_metrics_basic(self):
        """Test basic metrics computation."""
        evaluator = MultiClassEvaluator(class_names=self.class_names)
        metrics = evaluator.compute_metrics(self.y_true, self.y_pred)

        assert isinstance(metrics, MultiClassMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.macro_precision <= 1
        assert 0 <= metrics.macro_recall <= 1
        assert 0 <= metrics.macro_f1 <= 1
        assert len(metrics.per_class_precision) == self.n_classes
        assert len(metrics.per_class_recall) == self.n_classes
        assert len(metrics.per_class_f1) == self.n_classes
        assert metrics.confusion_matrix.shape == (self.n_classes, self.n_classes)

    def test_compute_metrics_with_probabilities(self):
        """Test metrics computation with probabilities."""
        evaluator = MultiClassEvaluator(class_names=self.class_names)
        metrics = evaluator.compute_metrics(self.y_true, self.y_pred, self.y_proba)

        assert isinstance(metrics, MultiClassMetrics)
        # ROC-AUC and average precision should be computed when probabilities are provided
        assert metrics.roc_auc_ovo is not None
        assert metrics.roc_auc_ovr is not None
        assert metrics.average_precision is not None
        assert 0 <= metrics.roc_auc_ovo <= 1
        assert 0 <= metrics.roc_auc_ovr <= 1
        assert 0 <= metrics.average_precision <= 1

    def test_compute_metrics_without_class_names(self):
        """Test metrics computation without providing class names."""
        evaluator = MultiClassEvaluator()
        metrics = evaluator.compute_metrics(self.y_true, self.y_pred)

        assert isinstance(metrics, MultiClassMetrics)
        # Should generate default class names
        expected_names = [f"RC-{i + 1:02d}" for i in range(self.n_classes)]
        assert all(name in metrics.per_class_precision for name in expected_names)

    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        evaluator = MultiClassEvaluator(class_names=self.class_names)
        metrics = evaluator.compute_metrics(self.y_true, self.y_pred)

        # Test count confusion matrix
        fig_count = evaluator.plot_confusion_matrix(
            metrics, title="Test Confusion Matrix", normalize=False
        )
        assert fig_count is not None

        # Test normalized confusion matrix
        fig_norm = evaluator.plot_confusion_matrix(
            metrics, title="Test Normalized Confusion Matrix", normalize=True
        )
        assert fig_norm is not None

        # Test saving to file
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "confusion_matrix.png"
            fig = evaluator.plot_confusion_matrix(
                metrics, save_path=save_path, normalize=False
            )
            assert save_path.exists()

    def test_generate_classification_report(self):
        """Test classification report generation."""
        evaluator = MultiClassEvaluator(class_names=self.class_names)
        metrics = evaluator.compute_metrics(self.y_true, self.y_pred)

        report = evaluator.generate_classification_report(metrics)
        assert isinstance(report, str)
        assert "Classification Report" in report
        assert "Accuracy" in report
        assert "Macro Precision" in report
        assert "Macro Recall" in report
        assert "Macro F1-Score" in report

        # Test saving to file
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "report.txt"
            evaluator.generate_classification_report(metrics, save_path=save_path)
            assert save_path.exists()
            with open(save_path, "r") as f:
                saved_report = f.read()
            assert "Classification Report" in saved_report

    def test_save_all_metrics(self):
        """Test saving all metrics to files."""
        evaluator = MultiClassEvaluator(class_names=self.class_names)
        metrics = evaluator.compute_metrics(self.y_true, self.y_pred, self.y_proba)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "evaluation_output"
            saved_files = evaluator.save_all_metrics(metrics, output_dir, prefix="test")

            expected_files = [
                "json_metrics",
                "classification_report",
                "per_class_csv",
                "confusion_matrix_count",
                "confusion_matrix_normalized",
            ]

            for file_type in expected_files:
                assert file_type in saved_files
                assert saved_files[file_type].exists()

            # Verify JSON file content
            with open(saved_files["json_metrics"], "r") as f:
                json_data = json.load(f)
            assert "accuracy" in json_data
            assert "macro_precision" in json_data

            # Verify CSV file content
            df = pd.read_csv(saved_files["per_class_csv"], index_col=0)
            assert len(df) == self.n_classes
            assert "precision" in df.columns
            assert "recall" in df.columns

    def test_perfect_classification(self):
        """Test metrics for perfect classification."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])  # Perfect predictions

        evaluator = MultiClassEvaluator(class_names=["A", "B", "C"])
        metrics = evaluator.compute_metrics(y_true, y_pred)

        assert metrics.accuracy == 1.0
        assert metrics.macro_precision == 1.0
        assert metrics.macro_recall == 1.0
        assert metrics.macro_f1 == 1.0
        assert np.all(np.diag(metrics.confusion_matrix) == [2, 2, 2])

    def test_random_classification(self):
        """Test metrics for random classification."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 2, 0, 2, 0, 1])  # All wrong predictions

        evaluator = MultiClassEvaluator(class_names=["A", "B", "C"])
        metrics = evaluator.compute_metrics(y_true, y_pred)

        assert metrics.accuracy == 0.0
        # With zero_division=0, precision/recall/F1 should be 0 for all classes
        assert metrics.macro_precision == 0.0
        assert metrics.macro_recall == 0.0
        assert metrics.macro_f1 == 0.0

    def test_single_class(self):
        """Test metrics for single class classification."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])

        evaluator = MultiClassEvaluator(class_names=["SingleClass"])
        metrics = evaluator.compute_metrics(y_true, y_pred)

        assert metrics.accuracy == 1.0
        assert metrics.macro_precision == 1.0
        assert metrics.macro_recall == 1.0
        assert metrics.macro_f1 == 1.0
        assert metrics.confusion_matrix.shape == (1, 1)
        assert metrics.confusion_matrix[0, 0] == 5

    def test_imbalanced_classes(self):
        """Test metrics with imbalanced class distribution."""
        # Highly imbalanced: 90 samples of class 0, 10 of class 1
        y_true = np.array([0] * 90 + [1] * 10)
        # Predict all as majority class
        y_pred = np.array([0] * 100)

        evaluator = MultiClassEvaluator(class_names=["Majority", "Minority"])
        metrics = evaluator.compute_metrics(y_true, y_pred)

        # Accuracy should be high (90%) but recall for minority class is 0
        assert metrics.accuracy == 0.9
        assert metrics.per_class_recall["Minority"] == 0.0
        assert (
            metrics.per_class_precision["Minority"] == 0.0
        )  # No predictions for minority class

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty arrays
        y_true = np.array([])
        y_pred = np.array([])

        evaluator = MultiClassEvaluator()
        with pytest.raises(ValueError):
            evaluator.compute_metrics(y_true, y_pred)

        # Single sample
        y_true = np.array([0])
        y_pred = np.array([0])
        metrics = evaluator.compute_metrics(y_true, y_pred)
        assert metrics.accuracy == 1.0

        # Mismatched lengths
        y_true = np.array([0, 1])
        y_pred = np.array([0])
        with pytest.raises(ValueError):
            evaluator.compute_metrics(y_true, y_pred)
