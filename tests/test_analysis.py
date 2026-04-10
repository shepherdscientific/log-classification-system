"""
Unit tests for root cause analysis module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, MagicMock

from src.evaluation.analysis import (
    RootCauseAnalyzer,
    MisclassificationAnalysis,
    FeatureImportanceAnalysis,
    RootCauseInsights,
)


class TestMisclassificationAnalysis:
    """Test MisclassificationAnalysis dataclass."""

    def test_analysis_creation(self):
        """Test creating MisclassificationAnalysis object."""
        cm = np.array([[8, 2], [1, 9]])
        class_names = ["RC-01", "RC-02"]

        analysis = MisclassificationAnalysis(
            confusion_matrix=cm,
            class_names=class_names,
            top_misclassifications=[
                {
                    "true_class": "RC-01",
                    "predicted_class": "RC-02",
                    "count": 2,
                    "percentage": 0.2,
                }
            ],
            per_class_error_rate={"RC-01": 0.2, "RC-02": 0.1},
            challenging_pairs=[
                {
                    "pair": "RC-01 → RC-02",
                    "count": 2,
                    "error_rate": 0.2,
                    "description": "RC-01 frequently misclassified as RC-02",
                }
            ],
        )

        assert analysis.confusion_matrix.shape == (2, 2)
        assert analysis.class_names == ["RC-01", "RC-02"]
        assert len(analysis.top_misclassifications) == 1
        assert analysis.per_class_error_rate["RC-01"] == 0.2
        assert len(analysis.challenging_pairs) == 1

    def test_to_dict(self):
        """Test converting analysis to dictionary."""
        cm = np.array([[5, 0], [0, 5]])
        analysis = MisclassificationAnalysis(
            confusion_matrix=cm,
            class_names=["A", "B"],
            top_misclassifications=[],
            per_class_error_rate={"A": 0.0, "B": 0.0},
            challenging_pairs=[],
        )

        result = analysis.to_dict()
        assert "confusion_matrix" in result
        assert "class_names" in result
        assert "top_misclassifications" in result
        assert "per_class_error_rate" in result
        assert "challenging_pairs" in result
        assert result["class_names"] == ["A", "B"]


class TestFeatureImportanceAnalysis:
    """Test FeatureImportanceAnalysis dataclass."""

    def test_analysis_creation(self):
        """Test creating FeatureImportanceAnalysis object."""
        analysis = FeatureImportanceAnalysis(
            feature_names=["feature1", "feature2", "feature3"],
            global_importance={"feature1": 0.5, "feature2": 0.3, "feature3": 0.2},
            per_class_importance={
                "RC-01": {"feature1": 0.6, "feature2": 0.3, "feature3": 0.1},
                "RC-02": {"feature1": 0.4, "feature2": 0.4, "feature3": 0.2},
            },
            top_features_per_class={
                "RC-01": ["feature1", "feature2"],
                "RC-02": ["feature2", "feature1"],
            },
            common_important_features=["feature1", "feature2"],
        )

        assert len(analysis.feature_names) == 3
        assert analysis.global_importance["feature1"] == 0.5
        assert "RC-01" in analysis.per_class_importance
        assert len(analysis.top_features_per_class["RC-01"]) == 2
        assert "feature1" in analysis.common_important_features

    def test_to_dict(self):
        """Test converting analysis to dictionary."""
        analysis = FeatureImportanceAnalysis(
            feature_names=["f1", "f2"],
            global_importance={"f1": 0.6, "f2": 0.4},
            per_class_importance={"Class1": {"f1": 0.7, "f2": 0.3}},
            top_features_per_class={"Class1": ["f1", "f2"]},
            common_important_features=["f1"],
        )

        result = analysis.to_dict()
        assert "feature_names" in result
        assert "global_importance" in result
        assert "per_class_importance" in result
        assert "top_features_per_class" in result
        assert "common_important_features" in result
        assert result["feature_names"] == ["f1", "f2"]


class TestRootCauseInsights:
    """Test RootCauseInsights dataclass."""

    def test_insights_creation(self):
        """Test creating RootCauseInsights object."""
        # Create mock analyses
        misclassification_analysis = MisclassificationAnalysis(
            confusion_matrix=np.array([[5, 0], [0, 5]]),
            class_names=["RC-01", "RC-02"],
            top_misclassifications=[],
            per_class_error_rate={"RC-01": 0.0, "RC-02": 0.0},
            challenging_pairs=[],
        )

        feature_importance_analysis = FeatureImportanceAnalysis(
            feature_names=["feature1"],
            global_importance={"feature1": 1.0},
            per_class_importance={
                "RC-01": {"feature1": 1.0},
                "RC-02": {"feature1": 1.0},
            },
            top_features_per_class={"RC-01": ["feature1"], "RC-02": ["feature1"]},
            common_important_features=["feature1"],
        )

        insights = RootCauseInsights(
            misclassification_analysis=misclassification_analysis,
            feature_importance_analysis=feature_importance_analysis,
            overall_accuracy=1.0,
            most_challenging_class="RC-01",
            easiest_class="RC-02",
            recommendations=[
                {
                    "type": "test",
                    "priority": "low",
                    "description": "Test",
                    "action": "Test",
                }
            ],
            identified_patterns=[
                {"type": "test", "description": "Test", "implication": "Test"}
            ],
        )

        assert insights.overall_accuracy == 1.0
        assert insights.most_challenging_class == "RC-01"
        assert insights.easiest_class == "RC-02"
        assert len(insights.recommendations) == 1
        assert len(insights.identified_patterns) == 1

    def test_to_dict(self):
        """Test converting insights to dictionary."""
        misclassification_analysis = MisclassificationAnalysis(
            confusion_matrix=np.array([[1]]),
            class_names=["RC-01"],
            top_misclassifications=[],
            per_class_error_rate={"RC-01": 0.0},
            challenging_pairs=[],
        )

        feature_importance_analysis = FeatureImportanceAnalysis(
            feature_names=["f1"],
            global_importance={"f1": 1.0},
            per_class_importance={"RC-01": {"f1": 1.0}},
            top_features_per_class={"RC-01": ["f1"]},
            common_important_features=["f1"],
        )

        insights = RootCauseInsights(
            misclassification_analysis=misclassification_analysis,
            feature_importance_analysis=feature_importance_analysis,
            overall_accuracy=1.0,
            most_challenging_class="RC-01",
            easiest_class="RC-01",
            recommendations=[],
            identified_patterns=[],
        )

        result = insights.to_dict()
        assert "overall_accuracy" in result
        assert "most_challenging_class" in result
        assert "recommendations" in result
        assert "identified_patterns" in result
        assert "misclassification_analysis" in result
        assert "feature_importance_analysis" in result

    def test_save(self):
        """Test saving insights to file."""
        misclassification_analysis = MisclassificationAnalysis(
            confusion_matrix=np.array([[1]]),
            class_names=["RC-01"],
            top_misclassifications=[],
            per_class_error_rate={"RC-01": 0.0},
            challenging_pairs=[],
        )

        feature_importance_analysis = FeatureImportanceAnalysis(
            feature_names=["f1"],
            global_importance={"f1": 1.0},
            per_class_importance={"RC-01": {"f1": 1.0}},
            top_features_per_class={"RC-01": ["f1"]},
            common_important_features=["f1"],
        )

        insights = RootCauseInsights(
            misclassification_analysis=misclassification_analysis,
            feature_importance_analysis=feature_importance_analysis,
            overall_accuracy=1.0,
            most_challenging_class="RC-01",
            easiest_class="RC-01",
            recommendations=[],
            identified_patterns=[],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            insights.save(temp_path)

            with open(temp_path, "r") as f:
                loaded = json.load(f)

            assert loaded["overall_accuracy"] == 1.0
            assert loaded["most_challenging_class"] == "RC-01"
        finally:
            Path(temp_path).unlink()


class TestRootCauseAnalyzer:
    """Test RootCauseAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test initializing RootCauseAnalyzer."""
        analyzer = RootCauseAnalyzer(class_names=["RC-01", "RC-02", "RC-03"])
        assert analyzer.class_names == ["RC-01", "RC-02", "RC-03"]

        analyzer_default = RootCauseAnalyzer()
        assert analyzer_default.class_names is None

    def test_analyze_misclassifications_basic(self):
        """Test basic misclassification analysis."""
        analyzer = RootCauseAnalyzer(class_names=["RC-01", "RC-02"])

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5 each
        y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])  # 3 errors

        analysis = analyzer.analyze_misclassifications(y_true, y_pred)

        assert analysis.confusion_matrix.shape == (2, 2)
        assert analysis.class_names == ["RC-01", "RC-02"]
        assert len(analysis.top_misclassifications) > 0
        assert "RC-01" in analysis.per_class_error_rate
        assert "RC-02" in analysis.per_class_error_rate

    def test_analyze_misclassifications_with_sample_data(self):
        """Test misclassification analysis with sample data."""
        analyzer = RootCauseAnalyzer(class_names=["RC-01", "RC-02"])

        y_true = np.array([0, 1])
        y_pred = np.array([1, 0])  # Both wrong

        sample_data = pd.DataFrame(
            {
                "log_message": ["Error in service A", "Timeout in service B"],
                "service": ["service-a", "service-b"],
            }
        )

        analysis = analyzer.analyze_misclassifications(
            y_true, y_pred, sample_data=sample_data
        )

        assert analysis.misclassified_samples is not None
        assert len(analysis.misclassified_samples) == 2
        assert "true_label" in analysis.misclassified_samples.columns
        assert "predicted_label" in analysis.misclassified_samples.columns

    def test_analyze_feature_importance_with_mock_model(self):
        """Test feature importance analysis with mock model."""
        analyzer = RootCauseAnalyzer(class_names=["RC-01", "RC-02"])

        # Create mock model with feature_importances_
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.6, 0.4])
        mock_model.predict_proba = Mock(return_value=np.array([[0.7, 0.3], [0.4, 0.6]]))

        X = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        y_true = np.array([0, 1])

        analysis = analyzer.analyze_feature_importance(
            mock_model, X, y_true, feature_names=["feature1", "feature2"]
        )

        assert len(analysis.feature_names) == 2
        assert "feature1" in analysis.global_importance
        assert "RC-01" in analysis.per_class_importance
        assert "RC-02" in analysis.per_class_importance
        assert "RC-01" in analysis.top_features_per_class

    def test_generate_insights(self):
        """Test generating comprehensive insights."""
        analyzer = RootCauseAnalyzer(class_names=["RC-01", "RC-02"])

        # Create mock model
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.6, 0.4])

        # Test data
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])  # 2 errors

        X = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5, 6], "feature2": [2, 3, 4, 5, 6, 7]}
        )

        sample_data = pd.DataFrame(
            {
                "log_message": [
                    "Error A",
                    "Error B",
                    "Error C",
                    "Error D",
                    "Error E",
                    "Error F",
                ]
            }
        )

        insights = analyzer.generate_insights(
            y_true, y_pred, mock_model, X, sample_data
        )

        assert insights.overall_accuracy == 4 / 6  # 4 correct out of 6
        assert insights.most_challenging_class in ["RC-01", "RC-02"]
        assert insights.easiest_class in ["RC-01", "RC-02"]
        assert len(insights.recommendations) > 0
        assert len(insights.identified_patterns) >= 0

    def test_generate_recommendations(self):
        """Test recommendation generation logic."""
        analyzer = RootCauseAnalyzer(class_names=["RC-01", "RC-02"])

        # Create mock analyses
        misclassification_analysis = MisclassificationAnalysis(
            confusion_matrix=np.array(
                [[3, 2], [1, 4]]
            ),  # RC-01: 2/5 wrong, RC-02: 1/5 wrong
            class_names=["RC-01", "RC-02"],
            top_misclassifications=[
                {
                    "true_class": "RC-01",
                    "predicted_class": "RC-02",
                    "count": 2,
                    "percentage": 0.4,
                }
            ],
            per_class_error_rate={"RC-01": 0.4, "RC-02": 0.2},
            challenging_pairs=[
                {
                    "pair": "RC-01 → RC-02",
                    "count": 2,
                    "error_rate": 0.4,
                    "description": "RC-01 frequently misclassified as RC-02",
                }
            ],
        )

        feature_importance_analysis = FeatureImportanceAnalysis(
            feature_names=["f1", "f2", "f3"],
            global_importance={"f1": 0.5, "f2": 0.3, "f3": 0.2},
            per_class_importance={
                "RC-01": {"f1": 0.6, "f2": 0.3, "f3": 0.1},
                "RC-02": {"f1": 0.4, "f2": 0.4, "f3": 0.2},
            },
            top_features_per_class={
                "RC-01": ["f1", "f2", "f3"],
                "RC-02": ["f2", "f1", "f3"],
            },
            common_important_features=["f1", "f2"],
        )

        # Test with low accuracy
        recommendations = analyzer._generate_recommendations(
            misclassification_analysis, feature_importance_analysis, 0.65
        )

        assert len(recommendations) > 0
        assert any(rec["type"] == "model_improvement" for rec in recommendations)
        assert any(rec["type"] == "class_specific" for rec in recommendations)
        assert any("RC-01" in rec["description"] for rec in recommendations)

    def test_identify_patterns(self):
        """Test pattern identification logic."""
        analyzer = RootCauseAnalyzer(class_names=["RC-01", "RC-02", "RC-03"])

        # Create mock analyses
        misclassification_analysis = MisclassificationAnalysis(
            confusion_matrix=np.array([[5, 2, 0], [1, 4, 0], [0, 0, 5]]),
            class_names=["RC-01", "RC-02", "RC-03"],
            top_misclassifications=[
                {
                    "true_class": "RC-01",
                    "predicted_class": "RC-02",
                    "count": 2,
                    "percentage": 0.29,
                },
                {
                    "true_class": "RC-02",
                    "predicted_class": "RC-01",
                    "count": 1,
                    "percentage": 0.2,
                },
            ],
            per_class_error_rate={"RC-01": 0.29, "RC-02": 0.2, "RC-03": 0.0},
            challenging_pairs=[],
        )

        feature_importance_analysis = FeatureImportanceAnalysis(
            feature_names=["f1", "f2"],
            global_importance={"f1": 0.8, "f2": 0.2},
            per_class_importance={
                "RC-01": {"f1": 0.9, "f2": 0.1},
                "RC-02": {"f1": 0.7, "f2": 0.3},
                "RC-03": {"f1": 0.8, "f2": 0.2},
            },
            top_features_per_class={
                "RC-01": ["f1", "f2"],
                "RC-02": ["f1", "f2"],
                "RC-03": ["f1", "f2"],
            },
            common_important_features=["f1"],
        )

        patterns = analyzer._identify_patterns(
            misclassification_analysis, feature_importance_analysis
        )

        assert len(patterns) > 0
        pattern_types = [p["type"] for p in patterns]
        assert any(
            t in pattern_types
            for t in ["error_clustering", "asymmetric_confusion", "feature_dominance"]
        )
