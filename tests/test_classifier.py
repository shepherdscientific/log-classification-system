"""
Unit tests for LogClassifier.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import joblib

from src.models.classifier import LogClassifier, ModelConfig


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ModelConfig()
        assert config.model_type == "random_forest"
        assert config.random_state == 42
        assert config.class_weight == "balanced"
        assert config.tune_hyperparameters is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            model_type="logistic_regression",
            random_state=123,
            class_weight=None,
            tune_hyperparameters=False,
        )
        assert config.model_type == "logistic_regression"
        assert config.random_state == 123
        assert config.class_weight is None
        assert config.tune_hyperparameters is False


class TestLogClassifier:
    """Test LogClassifier class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)

        # Create synthetic data with 8 classes
        n_samples = 100
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        y = np.random.choice(range(8), size=n_samples)

        feature_names = [f"feature_{i}" for i in range(n_features)]

        return X, y, feature_names

    def test_init_default(self):
        """Test initialization with default config."""
        classifier = LogClassifier()
        assert classifier.config.model_type == "random_forest"
        assert classifier.model is None
        assert classifier.best_params_ is None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ModelConfig(model_type="logistic_regression")
        classifier = LogClassifier(config=config)
        assert classifier.config.model_type == "logistic_regression"

    def test_create_model_logistic_regression(self):
        """Test creating logistic regression model."""
        config = ModelConfig(model_type="logistic_regression")
        classifier = LogClassifier(config=config)
        model = classifier._create_model()

        from sklearn.linear_model import LogisticRegression

        assert isinstance(model, LogisticRegression)
        assert model.C == 1.0
        assert model.class_weight == "balanced"

    def test_create_model_random_forest(self):
        """Test creating random forest model."""
        config = ModelConfig(model_type="random_forest")
        classifier = LogClassifier(config=config)
        model = classifier._create_model()

        from sklearn.ensemble import RandomForestClassifier

        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 100
        assert model.class_weight == "balanced"

    def test_create_model_xgboost(self, monkeypatch):
        """Test creating XGBoost model."""
        # Mock xgboost import by removing the module
        import sys

        if "xgboost" in sys.modules:
            monkeypatch.delitem(sys.modules, "xgboost", raising=False)

        config = ModelConfig(model_type="xgboost")
        classifier = LogClassifier(config=config)

        # Should raise ImportError
        with pytest.raises(ImportError, match="XGBoost not installed"):
            classifier._create_model()

    def test_create_model_invalid_type(self):
        """Test creating model with invalid type."""
        config = ModelConfig(model_type="invalid_model")
        classifier = LogClassifier(config=config)

        with pytest.raises(ValueError, match="Unknown model type"):
            classifier._create_model()

    def test_get_hyperparameter_grid_logistic(self):
        """Test hyperparameter grid for logistic regression."""
        config = ModelConfig(model_type="logistic_regression")
        classifier = LogClassifier(config=config)
        grid = classifier._get_hyperparameter_grid()

        assert "C" in grid
        assert "solver" in grid
        assert "class_weight" in grid
        assert len(grid["C"]) == 4

    def test_get_hyperparameter_grid_random_forest(self):
        """Test hyperparameter grid for random forest."""
        config = ModelConfig(model_type="random_forest")
        classifier = LogClassifier(config=config)
        grid = classifier._get_hyperparameter_grid()

        assert "n_estimators" in grid
        assert "max_depth" in grid
        assert "class_weight" in grid

    def test_fit_without_tuning(self, sample_data):
        """Test fitting model without hyperparameter tuning."""
        X, y, feature_names = sample_data

        config = ModelConfig(model_type="random_forest", tune_hyperparameters=False)
        classifier = LogClassifier(config=config)

        # Fit model
        classifier.fit(X, y, feature_names=feature_names)

        assert classifier.model is not None
        assert classifier.classes_ is not None
        assert len(classifier.classes_) == 8
        assert classifier.feature_names_ == feature_names
        assert classifier.best_params_ is None

    def test_fit_with_tuning(self, sample_data):
        """Test fitting model with hyperparameter tuning."""
        X, y, feature_names = sample_data

        config = ModelConfig(
            model_type="random_forest",
            tune_hyperparameters=True,
            cv_folds=3,  # Small CV for test
        )
        classifier = LogClassifier(config=config)

        # Fit model
        classifier.fit(X, y, feature_names=feature_names)

        assert classifier.model is not None
        assert classifier.best_params_ is not None
        assert classifier.cv_results_ is not None
        assert "best_score" in classifier.cv_results_

    def test_predict(self, sample_data):
        """Test prediction."""
        X, y, feature_names = sample_data

        classifier = LogClassifier()
        classifier.fit(X, y, feature_names=feature_names)

        predictions = classifier.predict(X[:5])
        assert predictions.shape == (5,)
        assert predictions.dtype == np.dtype("int64")

    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y, feature_names = sample_data

        classifier = LogClassifier()
        classifier.fit(X, y, feature_names=feature_names)

        probabilities = classifier.predict_proba(X[:5])
        assert probabilities.shape == (5, 8)  # 5 samples, 8 classes
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-10)

    def test_predict_before_fit(self):
        """Test prediction before fitting raises error."""
        classifier = LogClassifier()
        X = np.random.randn(5, 10)

        with pytest.raises(ValueError, match="Model not trained"):
            classifier.predict(X)

    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        X, y, feature_names = sample_data

        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        classifier = LogClassifier()
        classifier.fit(X_train, y_train, feature_names=feature_names)

        # Get unique classes in test set
        test_classes = np.unique(y_test)
        target_names = [f"RC-{i + 1:02d}" for i in test_classes]

        metrics = classifier.evaluate(X_test, y_test, return_report=True)

        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics
        assert isinstance(metrics["accuracy"], float)
        assert metrics["accuracy"] >= 0.0 and metrics["accuracy"] <= 1.0

    def test_get_feature_importance_random_forest(self, sample_data):
        """Test feature importance for random forest."""
        X, y, feature_names = sample_data

        config = ModelConfig(model_type="random_forest")
        classifier = LogClassifier(config=config)
        classifier.fit(X, y, feature_names=feature_names)

        importance = classifier.get_feature_importance()
        assert importance is not None
        assert len(importance) == len(feature_names)

        # Check all features have importance scores
        for feature in feature_names:
            assert feature in importance
            assert isinstance(importance[feature], float)

    def test_get_feature_importance_logistic(self, sample_data):
        """Test feature importance for logistic regression."""
        X, y, feature_names = sample_data

        config = ModelConfig(model_type="logistic_regression")
        classifier = LogClassifier(config=config)
        classifier.fit(X, y, feature_names=feature_names)

        importance = classifier.get_feature_importance()
        assert importance is not None
        assert len(importance) == len(feature_names)

    def test_save_and_load(self, sample_data):
        """Test model serialization."""
        X, y, feature_names = sample_data

        # Create and fit model
        classifier = LogClassifier()
        classifier.fit(X, y, feature_names=feature_names)

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            classifier.save(tmp_path)
            assert Path(tmp_path).exists()

            # Load model
            loaded_classifier = LogClassifier.load(tmp_path)

            # Check loaded model
            assert loaded_classifier.model is not None
            assert loaded_classifier.config.model_type == "random_forest"
            assert np.array_equal(classifier.classes_, loaded_classifier.classes_)
            assert classifier.feature_names_ == loaded_classifier.feature_names_

            # Check predictions match
            predictions_original = classifier.predict(X[:5])
            predictions_loaded = loaded_classifier.predict(X[:5])
            assert np.array_equal(predictions_original, predictions_loaded)

        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_before_fit(self):
        """Test saving before fitting raises error."""
        classifier = LogClassifier()

        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            with pytest.raises(ValueError, match="Model not trained"):
                classifier.save(tmp.name)

    def test_get_config_summary(self, sample_data):
        """Test configuration summary."""
        X, y, feature_names = sample_data

        classifier = LogClassifier()
        classifier.fit(X, y, feature_names=feature_names)

        summary = classifier.get_config_summary()

        assert "model_type" in summary
        assert "random_state" in summary
        assert "classes" in summary
        assert "n_classes" in summary
        assert summary["n_classes"] == 8

    def test_fit_with_validation_data(self, sample_data):
        """Test fitting with validation data."""
        X, y, feature_names = sample_data

        # Split data
        split_idx = int(0.7 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        classifier = LogClassifier()
        classifier.fit(
            X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feature_names
        )

        assert classifier.model is not None
        # Note: validation_score_ is set internally but not part of public API

    def test_class_imbalance_handling(self):
        """Test class imbalance handling."""
        # Create imbalanced data
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        # Create imbalanced labels (class 0 has 80 samples, others have 5 each)
        y = np.zeros(n_samples)
        y[80:85] = 1
        y[85:90] = 2
        y[90:95] = 3
        y[95:] = 4

        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Test with class_weight="balanced"
        config = ModelConfig(
            model_type="random_forest",
            class_weight="balanced",
            tune_hyperparameters=False,  # Disable tuning for simpler test
        )
        classifier = LogClassifier(config=config)
        classifier.fit(X, y, feature_names=feature_names)

        assert classifier.model is not None
        # RandomForest with class_weight="balanced" sets computed weights on the model
        assert classifier.model.class_weight is not None
        # Check that weights are computed (should be a dict, not "balanced")
        assert isinstance(classifier.model.class_weight, dict)
        # Should have weights for all 5 classes
        assert len(classifier.model.class_weight) == 5

    def test_class_imbalance_custom_weights(self):
        """Test class imbalance with custom weights."""
        # Create imbalanced data
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])

        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Test with class_weight=None (should compute weights)
        config = ModelConfig(model_type="random_forest", class_weight=None)
        classifier = LogClassifier(config=config)
        classifier.fit(X, y, feature_names=feature_names)

        assert classifier.model is not None
        # RandomForest with class_weight=None doesn't compute weights automatically
        # but our implementation computes them separately
