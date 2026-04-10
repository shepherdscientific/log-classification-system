"""
Unit tests for inference pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from unittest.mock import Mock, patch

from src.inference.predictor import LogPredictor, PredictionResult
from src.inference.summary import RootCauseSummary


class TestPredictionResult:
    """Test PredictionResult dataclass."""

    def test_prediction_result_creation(self):
        """Test creating a PredictionResult."""
        top_predictions = [("RC-01", 0.8), ("RC-02", 0.15), ("RC-03", 0.05)]
        features_used = {
            "log_message": "Test message",
            "service": "test-service",
            "severity": "ERROR",
        }

        result = PredictionResult(
            root_cause="RC-01",
            confidence=0.8,
            top_n_predictions=top_predictions,
            features_used=features_used,
        )

        assert result.root_cause == "RC-01"
        assert result.confidence == 0.8
        assert result.top_n_predictions == top_predictions
        assert result.features_used == features_used

    def test_to_dict(self):
        """Test converting PredictionResult to dictionary."""
        top_predictions = [("RC-01", 0.8), ("RC-02", 0.15)]
        features_used = {"log_message": "Test"}

        result = PredictionResult(
            root_cause="RC-01",
            confidence=0.8,
            top_n_predictions=top_predictions,
            features_used=features_used,
        )

        result_dict = result.to_dict()

        assert result_dict["root_cause"] == "RC-01"
        assert result_dict["confidence"] == 0.8
        assert len(result_dict["top_n_predictions"]) == 2
        assert result_dict["top_n_predictions"][0]["root_cause"] == "RC-01"
        assert result_dict["top_n_predictions"][0]["confidence"] == 0.8
        assert result_dict["features_used"] == features_used

    def test_to_json(self):
        """Test converting PredictionResult to JSON."""
        result = PredictionResult(
            root_cause="RC-01",
            confidence=0.8,
            top_n_predictions=[("RC-01", 0.8)],
            features_used={"test": "value"},
        )

        json_str = result.to_json()
        assert "RC-01" in json_str
        assert "0.8" in json_str
        assert "test" in json_str


class TestLogPredictor:
    """Test LogPredictor class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with predict and predict_proba methods."""
        model = Mock()
        model.predict.return_value = np.array(["RC-01", "RC-02"])
        model.predict_proba.return_value = np.array(
            [
                [0.8, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.7, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0],
            ]
        )
        return model

    @pytest.fixture
    def mock_feature_engineer(self):
        """Create a mock feature engineer."""
        engineer = Mock()
        engineer.transform.return_value = pd.DataFrame({"feature": [1, 2]})
        # Add methods that the predictor expects
        engineer.preprocess_text = Mock(return_value="test message")
        engineer.extract_error_type = Mock(return_value="database_error")
        engineer.extract_service_patterns = Mock(return_value=["api", "gateway"])
        engineer.tfidf_vectorizer = Mock()
        # Return a sparse matrix instead of numpy array
        from scipy.sparse import csr_matrix

        engineer.tfidf_vectorizer.transform = Mock(
            return_value=csr_matrix(np.array([[0.1, 0.2, 0.3]]))
        )
        engineer.service_encoder = Mock()
        engineer.service_encoder.transform = Mock(return_value=np.array([0]))
        return engineer

    @pytest.fixture
    def sample_predictor(self, mock_model, mock_feature_engineer):
        """Create a LogPredictor with mocked dependencies."""
        with patch("joblib.load") as mock_load:
            mock_load.return_value = {
                "model": mock_model,
                "feature_engineer": mock_feature_engineer,
                "root_cause_labels": [f"RC-{i:02d}" for i in range(1, 9)],
            }

            predictor = LogPredictor("dummy_model.joblib")
            return predictor

    def test_predictor_initialization(self, sample_predictor):
        """Test predictor initialization."""
        assert sample_predictor.model is not None
        assert sample_predictor.feature_engineer is not None
        assert len(sample_predictor.root_cause_labels) == 8
        assert sample_predictor.root_cause_labels[0] == "RC-01"

    def test_predict_single(self, sample_predictor):
        """Test single prediction."""
        # Mock should return single prediction for single input
        sample_predictor.model.predict.return_value = np.array(["RC-01"])
        sample_predictor.model.predict_proba.return_value = np.array(
            [[0.8, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0]]
        )

        # Mock feature engineer methods
        sample_predictor.feature_engineer.preprocess_text.return_value = (
            "database connection failed"
        )
        sample_predictor.feature_engineer.extract_error_type.return_value = (
            "database_error"
        )
        sample_predictor.feature_engineer.extract_service_patterns.return_value = [
            "payment",
            "service",
        ]
        from scipy.sparse import csr_matrix

        sample_predictor.feature_engineer.tfidf_vectorizer.transform.return_value = (
            csr_matrix(np.array([[0.1, 0.2, 0.3]]))
        )
        sample_predictor.feature_engineer.service_encoder.transform.return_value = (
            np.array([0])
        )

        result = sample_predictor.predict_single(
            log_message="Database connection failed",
            service="payment-service",
            severity="ERROR",
        )

        assert isinstance(result, PredictionResult)
        assert result.root_cause == "RC-01"
        assert result.confidence == 0.8
        assert len(result.top_n_predictions) == 3  # Default top_n=3

    def test_predict_batch(self, sample_predictor):
        """Test batch prediction."""
        batch_data = pd.DataFrame(
            [
                {
                    "log_message": "Database connection failed",
                    "service": "payment-service",
                    "severity": "ERROR",
                    "timestamp": "2026-04-09T10:15:30",
                },
                {
                    "log_message": "Memory usage high",
                    "service": "monitoring-service",
                    "severity": "WARNING",
                    "timestamp": "2026-04-09T10:16:45",
                },
            ]
        )

        # Mock feature engineer methods for batch
        sample_predictor.feature_engineer.preprocess_text.side_effect = [
            "database connection failed",
            "memory usage high",
        ]
        sample_predictor.feature_engineer.extract_error_type.side_effect = [
            "database_error",
            "memory_error",
        ]
        sample_predictor.feature_engineer.extract_service_patterns.side_effect = [
            ["payment", "service"],
            ["monitoring", "service"],
        ]
        from scipy.sparse import csr_matrix

        sample_predictor.feature_engineer.tfidf_vectorizer.transform.return_value = (
            csr_matrix(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
        )
        sample_predictor.feature_engineer.service_encoder.transform.return_value = (
            np.array([0, 1])
        )

        results = sample_predictor.predict_batch(batch_data, top_n=2)

        assert len(results) == 2
        assert results[0].root_cause == "RC-01"
        assert results[1].root_cause == "RC-02"
        assert len(results[0].top_n_predictions) == 2  # top_n=2

    def test_validate_input_data_valid(self, sample_predictor):
        """Test validation with valid data."""
        valid_data = pd.DataFrame(
            [
                {
                    "log_message": "Test message",
                    "service": "test-service",
                    "severity": "ERROR",
                }
            ]
        )

        # Should not raise exception
        sample_predictor._validate_input_data(valid_data)

    def test_validate_input_data_missing_column(self, sample_predictor):
        """Test validation with missing required column."""
        invalid_data = pd.DataFrame(
            [
                {
                    "service": "test-service",  # Missing log_message
                    "severity": "ERROR",
                }
            ]
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            sample_predictor._validate_input_data(invalid_data)

    def test_validate_input_data_empty_message(self, sample_predictor):
        """Test validation with empty log message."""
        invalid_data = pd.DataFrame(
            [
                {
                    "log_message": "",  # Empty message
                    "service": "test-service",
                    "severity": "ERROR",
                }
            ]
        )

        with pytest.raises(ValueError, match="empty log messages"):
            sample_predictor._validate_input_data(invalid_data)

    def test_validate_input_data_nan_message(self, sample_predictor):
        """Test validation with NaN log message."""
        invalid_data = pd.DataFrame(
            [
                {
                    "log_message": np.nan,  # NaN message
                    "service": "test-service",
                    "severity": "ERROR",
                }
            ]
        )

        with pytest.raises(ValueError, match="empty log messages"):
            sample_predictor._validate_input_data(invalid_data)

    def test_validate_input_data_non_string_message(self, sample_predictor):
        """Test validation with non-string log message."""
        invalid_data = pd.DataFrame(
            [
                {
                    "log_message": 123,  # Non-string message
                    "service": "test-service",
                    "severity": "ERROR",
                }
            ]
        )

        with pytest.raises(ValueError, match="must contain strings"):
            sample_predictor._validate_input_data(invalid_data)

    def test_extract_features_used(self, sample_predictor):
        """Test feature extraction from log entry."""
        log_entry = pd.Series(
            {
                "log_message": "Test message",
                "service": "test-service",
                "severity": "ERROR",
                "timestamp": "2026-04-09T10:15:30",
            }
        )

        features = sample_predictor._extract_features_used(log_entry)

        assert features["log_message"] == "Test message"
        assert features["service"] == "test-service"
        assert features["severity"] == "ERROR"
        assert features["timestamp"] == "2026-04-09T10:15:30"

    def test_extract_features_used_with_nan(self, sample_predictor):
        """Test feature extraction with NaN values."""
        log_entry = pd.Series(
            {
                "log_message": "Test message",
                "service": np.nan,  # NaN value
                "severity": "ERROR",
                "timestamp": pd.NaT,  # NaT timestamp
            }
        )

        features = sample_predictor._extract_features_used(log_entry)

        assert features["log_message"] == "Test message"
        assert features["service"] == ""  # NaN converted to empty string
        assert features["severity"] == "ERROR"
        assert features["timestamp"] == ""  # NaT converted to empty string

    def test_model_without_predict_proba(self, mock_model, mock_feature_engineer):
        """Test prediction with model that doesn't have predict_proba."""
        # Create model without predict_proba
        del mock_model.predict_proba  # Remove predict_proba attribute
        mock_model.predict.return_value = np.array(["RC-01"])

        # Configure mock feature engineer
        mock_feature_engineer.preprocess_text.return_value = "test message"
        mock_feature_engineer.extract_error_type.return_value = "test_error"
        mock_feature_engineer.extract_service_patterns.return_value = [
            "test",
            "service",
        ]
        mock_feature_engineer.tfidf_vectorizer = Mock()
        from scipy.sparse import csr_matrix

        mock_feature_engineer.tfidf_vectorizer.transform.return_value = csr_matrix(
            np.array([[0.1, 0.2, 0.3]])
        )
        mock_feature_engineer.service_encoder = Mock()
        mock_feature_engineer.service_encoder.transform.return_value = np.array([0])

        with patch("joblib.load") as mock_load:
            mock_load.return_value = {
                "model": mock_model,
                "feature_engineer": mock_feature_engineer,
                "root_cause_labels": ["RC-01", "RC-02"],
            }

            predictor = LogPredictor("dummy_model.joblib")

            # Test prediction
            result = predictor.predict_single(
                log_message="Test message", service="test-service", severity="ERROR"
            )

            # Should still work, but confidence might be 1.0 or 0.0
            assert result.root_cause == "RC-01"

    def test_save_and_load(self, tmp_path):
        """Test saving and loading predictor."""
        # Create simple test objects that can be pickled
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Create simple model and feature engineer
        model = LogisticRegression(random_state=42)
        # Fit with dummy data
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        model.fit(X, y)

        feature_engineer = StandardScaler()
        feature_engineer.fit(X)

        root_cause_labels = ["RC-01", "RC-02"]

        # Create predictor with simple objects
        with patch("joblib.load") as mock_load:
            mock_load.return_value = {
                "model": model,
                "feature_engineer": feature_engineer,
                "root_cause_labels": root_cause_labels,
            }

            predictor = LogPredictor("dummy_model.joblib")

            # Save predictor
            save_path = tmp_path / "test_predictor.joblib"
            predictor.save(save_path)
            assert save_path.exists()

            # Test load class method
            with patch("joblib.load") as mock_load2:
                mock_load2.return_value = {
                    "model": model,
                    "feature_engineer": feature_engineer,
                    "root_cause_labels": root_cause_labels,
                    "model_path": "dummy_model.joblib",
                }

                loaded_predictor = LogPredictor.load(save_path)

                assert loaded_predictor.model is not None
                assert loaded_predictor.feature_engineer is not None
                assert loaded_predictor.root_cause_labels == root_cause_labels

    def test_prediction_with_summary(self, sample_predictor):
        """Test that predictions include summaries."""
        # Mock should return single prediction for single input
        sample_predictor.model.predict.return_value = np.array(["RC-01"])
        sample_predictor.model.predict_proba.return_value = np.array(
            [[0.8, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0]]
        )

        # Mock feature engineer methods
        sample_predictor.feature_engineer.preprocess_text.return_value = (
            "401 unauthorized invalid api key provided by client client_8478"
        )
        sample_predictor.feature_engineer.extract_error_type.return_value = (
            "authentication_error"
        )
        sample_predictor.feature_engineer.extract_service_patterns.return_value = [
            "api",
            "gateway",
        ]
        from scipy.sparse import csr_matrix

        sample_predictor.feature_engineer.tfidf_vectorizer.transform.return_value = (
            csr_matrix(np.array([[0.1, 0.2, 0.3]]))
        )
        sample_predictor.feature_engineer.service_encoder.transform.return_value = (
            np.array([0])
        )

        result = sample_predictor.predict_single(
            log_message="401 Unauthorized — invalid API key provided by client client_8478",
            service="api-gateway",
            severity="High",
        )

        assert isinstance(result, PredictionResult)
        assert result.root_cause == "RC-01"
        assert result.confidence == 0.8

        # Check that summary is generated
        assert result.summary is not None
        assert isinstance(result.summary, RootCauseSummary)
        assert result.summary.root_cause == "RC-01"
        assert result.summary.confidence == 0.8
        assert len(result.summary.key_evidence) > 0
        assert len(result.summary.recommended_actions) > 0

        # Check summary JSON serialization
        result_dict = result.to_dict()
        assert "summary" in result_dict
        assert result_dict["summary"]["root_cause"] == "RC-01"

    def test_prediction_result_generate_summary(self):
        """Test PredictionResult.generate_summary method."""
        result = PredictionResult(
            root_cause="RC-02",
            confidence=0.85,
            top_n_predictions=[("RC-02", 0.85), ("RC-01", 0.10)],
            features_used={"log_message": "Test"},
        )

        # Generate summary
        result_with_summary = result.generate_summary(
            log_message="Database connection pool exhausted",
            service="db-pool",
            severity="Critical",
            timestamp="2024-04-10T08:46:00Z",
        )

        assert result_with_summary.summary is not None
        assert result_with_summary.summary.root_cause == "RC-02"
        assert (
            "Database" in result_with_summary.summary.summary
            or "connection" in result_with_summary.summary.summary.lower()
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
